#!/bin/bash
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Program name	: DWARF_bench_automated.sh
# Purpose	: This script is written to use dwarf_bench to benchmark GPU/CPU
#		  It combines the work done by Petr A Kurapov on how to create a
#		  container for this purpose of benchmark and the work of
#		  Laurent Montigny on running dwarf_bench to benchmark GPU/CPU.
#		  This script automates the entire process that is:
# - Cleanup any previous container from previous run
# - git clone the repository create by Petr for this purpose
# - Create a docker image base on the Dockerfile 
# - Create a container with that image
# - Then set some env params in the container, build the dwarf_bench binary
# - Once the build is done, the command dwarf_bench can be used to benchmark the
#   CPU and GPU. It's where the script benchmarks.sh written by Laurent come to
#   take place.
# Author	: Mamadou Diatta - mamadou.diatta@intel.com
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Adding myself to the required groups docker render and video
user=$USER
WORKING_DIR=`pwd`
# Let us verify if $user is really added to the groups
echo 
echo "======= Let us verify if $user is really added to the groups ======="
echo
for group in docker render video
do
	echo "------- Group: $group"
	# Let us check if the group exist already
	getent group $group
	if [ $? != 0 ]
	then
		echo 
		echo "======= Creating the  group: $group ======="
		sudo groupadd $group
		if [ $? != 0 ]
		then
			echo
			echo
			echo "Creating the group: $group failed"
			echo "Please create it by hand before rerunning this script"
			echo
			echo
			exit 1
		else
			echo
			echo
			echo "Creating the group: $group is successful"
			echo
			echo
		fi
	else
		echo
		echo
		echo "The group: $group already exist in this host"
		echo
		echo
	fi
	id | grep $group | grep -v grep
	if [ $? != 0 ]
	then
		echo "Adding $user to group $group"
		sudo usermod -aG $group $user
		if [ $? !=0 ]
		then
			echo " Failed to add $user to group $group"
			exit 1
		else
			echo "$user is added to group $group successfully"
		fi
	else
		echo "$user is already added to group $group successfully"
	fi
done
# echo let us verify that I have access to the command docker
echo
echo "======= echo let us verify that I have access to the command docker ======="
echo
docker --version
if [ $? != 0 ]
then
	echo "You don't have access to docker or docker is not installed in this machine"
	exit 1
else
	echo "You have access to docker command in this host"
	echo
fi

# Let setup the proxies
echo
echo "======= Let us setup the proxies ======="
echo

export http_proxy="http://proxy-dmz.intel.com:912"
export https_proxy="http://proxy-dmz.intel.com:912"

env | grep proxy | grep -v grep
if [ $? != 0 ]
then
        echo "Failed to set up the proxies. I am out"
	echo
        exit 1
fi

# Let us remove any previously running or stopped container related to this:
echo "======= Let us remove any previously running or stopped container related to this ======"
CONTAINER_ID=`docker container ps -a | grep dwarfs-dev | grep -v grep | awk '{print $1}'`
if [ $CONTAINER_ID != "" ]
then
	echo "Running the command: docker container stop $CONTAINER_ID"
	docker container stop $CONTAINER_ID
	echo "Running the command: docker container rm -f $CONTAINER_ID"
	docker container rm -f $CONTAINER_ID
fi

# Let us remove any previously image realted to this"
echo "Let us remove any previously image realted to this"
echo
IMAGE_ID=`docker images | grep dwarfs-dev | grep -v grep | awk '{print $3}'`
if [ $IMAGE_ID != "" ]
then
	echo "Running the command: docker image rm -f $IMAGE_ID"
	docker image rm -f $IMAGE_ID
	if [ $? == 0 ]
	then
		echo "Running the command: docker image rm -f $IMAGE_ID is SUCCESSFUL"
		echo
	else
		echo "Running the command: docker image rm -f $IMAGE_ID FAILED"
		echo "Please try to remove the docker image: $IMAGE_ID yourself"
		echo
		exit 1
	fi
fi
# Let us git clone https://github.com/kurapov-peter/dwarf_bench.git

if [ -d dwarf_bench ]
then
	mv dwarf_bench dwarf_bench.$$
	if [ $? == 0 ]
	then
		echo "----- I move dwarf_bench to dwarf_bench.$$"
		ls -al dwarf_bench.$$
	else
		echo "Failed to move dwarf_bench to dwarf_bench.$$"
		echo "Please do it yourself move dwarf_bench out of here"
		echo "Restart this script: $0"
		echo
		exit 1
	fi
fi

echo "======= Let us git clone https://github.com/kurapov-peter/dwarf_bench.git ======"
echo
git clone https://github.com/kurapov-peter/dwarf_bench.git

if [ -d dwarf_bench ]
then
	cd dwarf_bench
	if [ -f Dockerfile ]
	then
		echo "======= cat Dockerfile ======"
		echo
		echo
		echo =========================================
		cat Dockerfile
		echo
		echo =========================================
		echo
		echo
		echo
		echo
		echo "======== Let go ahead and create the docker image ======"
		echo
		sudo docker build . --network host -t dwarfs-dev  \
		--build-arg http_proxy="http://Proxy-dmz.intel.com:911"  \
		--build-arg https_proxy="http://Proxy-dmz.intel.com:912" 
	
		if [ $? == 0 ]
		then
			echo
			echo
			echo
			echo "======= Let us go ahead and create the container ======="
			# sleep 30
			for i in `seq 1 15`; do echo -n "."; sleep 2; done
			echo
			docker run --network host --privileged -it -d \
			--name spicy -v $WORKING_DIR:/dwarf_bench \
			dwarfs-dev:latest bash
			#NOTE: Start the container in the background -d
			# The current directory `pwd` is mounted into
			# the container under /dwarf_bench
			# Let us go ahead and run the benchmarks.sh script
			# using /dwarf_bench/DWARF_bench_run_in_the_container.sh
			echo
			docker container ps -a | egrep "CONTAINER|spicy"
			echo
			echo
			echo
			echo
			echo "======= Running the benchmarks.sh ======="
			echo
			echo
			echo -n Please be patient . . . . . . .
			echo
			# sleep 120
			for i in `seq 1 30`; do echo -n "."; sleep 2; done
			echo
			docker exec -it spicy sh -x /dwarf_bench/DWARF_bench_run_in_the_container.sh
#			docker exec -it spicy /dwarf_bench/DWARF_bench_run_in_the_container-02.sh
			# sleep 120
			for i in `seq 1 30`; do echo -n "."; sleep 2; done
			echo
			echo
			echo
			echo "Please find the *csv files in this local host"
			echo 
			echo ============================================
			ls -al $WORKING_DIR/*.csv
			echo ============================================
			echo
			echo
		else
			echo "The command sudo docker build . FAILED"
			echo "
			-------------------------------------------------------
			Please run this command alone and see if
			this will work alone.
			sudo docker build . --network host -t dwarfs-dev  \
			--build-arg http_proxy=\"http://Proxy-dmz.intel.com:911\" \
			--build-arg https_proxy=\"http://Proxy-dmz.intel.com:912\"
			-------------------------------------------------------
			"
			echo
			echo
			exit 1
		fi	
	else
		echo "----- Dockerfile is missing in this directory"
		echo "Please make sure the Dockerfile is not missing in github"
		echo	
		exit 1
	fi
else
	echo "The directory dwarf_bench is missing from this git clone"
	echo "Please try to review it your repository if complete in github"
	exit 1
fi	
