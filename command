terminal 1
cd /home/jinbeibei/didi_ws/
source devel/setup.bash
roslaunch didi_tools play.launch 



terminal 2
cd /home/jinbeibei/didi_ws/
source devel/setup.bash
roslaunch didi_tools obs.launch bag:=/home/jinbeibei/File/Di-tech/data/car/training/bmw_following_long/bmw02.bag

