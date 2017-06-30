#include <ros/ros.h>  
#include <sensor_msgs/PointCloud2.h>

#include <boost/thread/thread.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>  
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>

#include <cv.h>
#include <highgui.h>

#include <stdio.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>

const int birdHeightMap = 8;
const double upBound = 1.0;
const double lowBound = -3.0;
const double mapDistance = (upBound - lowBound) / birdHeightMap;
const double maxDist = 50.0;
const double resolution = 0.1;
const double d_theta = 0.16; //degree
const double d_fai = 4.0 / 3;
const int birdX = maxDist / resolution * 2;
const int birdY = maxDist / resolution * 2;
const int frontX = 180 / d_theta;
const int frontY = 32;
const int backX = 180 / d_theta;
const int backY = 32;
const float LOG32 = 1.0 / log(32);


// string format1;
// pcl::PointCloud<pcl::PointXYZ>* cl;
boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cl(new pcl::PointCloud<pcl::PointXYZ>);
char pc_time[200];
void pc_vis(pcl::PointCloud<pcl::PointXYZ>::Ptr in);

// ensure: bird, front and back are NULL, because they are initialized in this function.
void projectPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr in, cv::Mat* &bird, cv::Mat* &front, cv::Mat* &back)
{
    pc_vis(in);
    bird = new cv::Mat[birdHeightMap+1];
    front = new cv::Mat[2];
    back = new cv::Mat[2];

    cv::Mat bird0 = cv::Mat::zeros(birdY, birdX, CV_32FC1);
    for(int i = 0; i < birdHeightMap + 1; i++)
    {
        bird[i] = cv::Mat::zeros(birdY, birdX, CV_8UC1);
    }
    for(int i=0;i<2;i++)
    {
        front[i]=cv::Mat::zeros(frontY, frontX, CV_8UC1);
        back[i]=cv::Mat::zeros(backY, backX, CV_8UC1);
    }

    for (unsigned int i = 0; i < in->points.size(); i++)
    {
        float x = in->points[i].x;
        float y = in->points[i].y;
        float z = in->points[i].z;
        if(x>maxDist || x<-maxDist || y>maxDist || y<-maxDist || z>upBound || z<lowBound)
            continue;
        int bx = (int)(birdX - (y + maxDist) / resolution + 1);
        int by = (int)(birdY - (x + maxDist) / resolution + 1);
        bx = bx < 1 ? 1 : (bx > birdX ? birdX : bx);
        by = by < 1 ? 1 : (by > birdY ? birdY : by);
        bird0.at<float>(by - 1,bx - 1) += 1.0; //height density
        int bz = (z - lowBound) / mapDistance;
        bird[bz + 1].at<uchar>(by - 1,bx - 1) = 255; //height slice
        if(x >= 0) //front map
        {
            int fx = (int)((90 - atan2(y, x) * 57.325) / d_theta + 1);
            int fy = (int)(9 - atan2(z, sqrt(x * x + y * y)) * 57.325 / d_fai);
            fx = fx < 1 ? 1 : (fx > frontX ? frontX : fx);
            fy = fy < 1 ? 1 : (fy > frontY ? frontY : fy);
            front[0].at<uchar>(fy-1,fx-1) = (uchar)((z - lowBound) / (upBound - lowBound));
            front[1].at<uchar>(fy-1,fx-1) = (uchar)(sqrt(x * x + y * y + z * z));
        }
        else //back map
        {
            int fx=(int)((90 - atan2(y, -x) * 57.325) / d_theta + 1);
            int fy=(int)(9 - atan2(z, sqrt(x * x + y * y)) * 57.325 / d_fai);
            fx = fx < 1 ? 1 : (fx > backX ?backX : fx);
            fy = fy < 1 ? 1 : (fy > backY ?backY : fy);
            back[0].at<uchar>(fy-1,fx-1) = (uchar)((z - lowBound) / (upBound - lowBound));
            back[1].at<uchar>(fy-1,fx-1) = (uchar)(sqrt(x * x + y * y + z * z));
        }
    }
    for(int r = 0; r < birdY; r++)
    {
        uchar* p1 = bird[0].ptr<uchar> (r);
        float* p2 = bird0.ptr<float> (r);
        for(int c = 0; c < birdX; c++)
        {
            float temp = log(p2[c] + 1) * LOG32;
            p1[0] = temp > 1.0 ? 255 : temp * 255;
        }
    }
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> customColourVis (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

void pc_vis(pcl::PointCloud<pcl::PointXYZ>::Ptr in)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = simpleVis(in);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
  return ;
}

void mv_vis(cv::Mat* bird, cv::Mat* front, cv::Mat* back)
{
  std::string fileorder[10] = {"1","2","3","4","5","6","7","8","9","10"};
  ROS_INFO("begin save");
  for (int i = 0; i < birdHeightMap + 1; ++i)
  {
    // cv::namedWindow("hehe"); 
    // cv::imshow("hehe", bird[i]);
    // cv::waitKey(0);
    std::string str = "/home/zym/testimg/" + fileorder[i] + ".jpg";
    ROS_INFO(str.c_str());
    cv::imwrite(str, bird[i]);
  }
  ROS_INFO("saved");
}

void pcCallback(const sensor_msgs::PointCloud2ConstPtr& msg)  
{
  // if(access(format1.c_str(),F_OK)!=0)
    // mkdir(format1.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  // sprintf(pc_time,"%10d%09d",(msg->header).stamp.sec,(msg->header).stamp.nsec);
  // cout<<pc_time<<endl;

  //pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;   
  //pcl_conversions::toPCL(*msg, *cloud);
  //cl=new pcl::PointCloud<pcl::PointXYZ>;
  //pcl::fromPCLPointCloud2(*cloud,*cl);
  pcl::fromROSMsg(*msg, *cl);
  ROS_INFO("get cl");
  // pc_vis(cl);
  cv::Mat* biredviews = NULL;
  cv::Mat* frontviews = NULL;
  cv::Mat* backviews = NULL;
  projectPCL(cl, biredviews, frontviews, backviews);
  ROS_INFO("get views");
  mv_vis(biredviews, frontviews, backviews);

  // char name1[200];
  // sprintf(name1,"%s/%s.bin",format1.c_str(),pc_time);
  // ofstream fout(name1);
  // for(int i=0;i<cl->points.size();i++){
  // 	if(abs(cl->points[i].x)<2.5&&abs(cl->points[i].y)<0.8)
  // 		continue;
  // 	fout<<cl->points[i].x<<" "<<cl->points[i].y<<" "<<cl->points[i].z<<endl;
  // }
  // fout.close();
  exit(0);
}

void detectCar(cv::Mat* bird, cv::Mat* front, cv::Mat* back)
{
  return;
}

int main(int argc, char **argv)  
{  
  // cl = new pcl::PointCloud<pcl::PointXYZ>;
  ros::init(argc, argv, "detect_car");  
  ros::NodeHandle nh_;  
  // ros::param::get("~format1",format1);
  int i = 0;
  ros::Subscriber pc_sub = nh_.subscribe("/velodyne_points", 10000, pcCallback);
  ros::spin();  
  return 0; 
}