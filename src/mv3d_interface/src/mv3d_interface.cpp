#include "api/api.hpp"
//#include "api/FRCNN/frcnn_api.hpp"
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


#include <vector>
#include "caffe/mv3d/util/frcnn_param.hpp"
#include <gflags/gflags.h>
//#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/mv3d/util/frcnn_vis.hpp"
#include<time.h>

#include "caffe/mv3d/util/frcnn_utils.hpp"

const int birdHeightMap = 8;
const double upBound = 1.0;
const double lowBound = -3.0;
const double mapDistance = (upBound - lowBound) / birdHeightMap;
const double maxDist = 50.0;
const double resolution = 0.1;
const double d_theta = 0.16; //degree
const double d_fai = 4.0 / 3;

const int birdX = maxDist / resolution /8;
const int birdY = maxDist / resolution /8;
const int frontX = 180 / d_theta;
const int frontY = 32;
const int backX = 180 / d_theta;
const int backY = 32;
const float LOG32 = 1.0 / log(32);


using caffe::Frcnn::FrcnnParam;
using namespace std;

// string format1;
// pcl::PointCloud<pcl::PointXYZ>* cl;
boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cl(new pcl::PointCloud<pcl::PointXYZ>);
char pc_time[200];
void pc_vis(pcl::PointCloud<pcl::PointXYZ>::Ptr in);

// ensure: bird, front and back are NULL, because they are initialized in this function.
void projectPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr in, cv::Mat* &bird, cv::Mat* &front, cv::Mat* &back)
{
	 
    //pc_vis(in);
    ROS_INFO("qqqqqq");
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
	 ROS_INFO("555555");
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
    ROS_INFO("wwwwwww");
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
    ROS_INFO("666666");
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
    ROS_INFO("ffffff");
    viewer->spinOnce (100);
	ROS_INFO("gggggg");
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    
  }
  ROS_INFO("rrrrrr");
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
    std::string str = "/home/jinbeibei/File/test/test" + fileorder[i] + ".jpg";
    ROS_INFO(str.c_str());
    cv::imwrite(str, bird[i]);
  }
  ROS_INFO("saved");
}


void detectCar(cv::Mat* bird, cv::Mat* front, cv::Mat* back,std::vector<std::vector<float> > &results_corners,std::vector<float> &score)
{
	int index_bird=0,index_front=0,index_back=0;
	

	vector<cv::Mat> bv_image;
	int bv_channels = caffe::Frcnn::FrcnnParam::bv_channels;
	
	int len_bird=sizeof(bird)/sizeof(bird[0]);
	int len_front=sizeof(bird)/sizeof(front[0]);
	int len_back=sizeof(bird)/sizeof(back[0]);


	//caffe::GlobalInit(&argc, &argv);  
	int gpu_id = 0;
	caffe::Caffe::set_mode(caffe::Caffe::GPU);		
	std::string proto_file             = "/home/jinbeibei/mv3d/models/FRCNN/mv3d/vgg16/test.proto";
	std::string model_file             = "/home/jinbeibei/mv3d/models/FRCNN/snapshot/vgg16_mv3d_bv_loss_iter_4000.caffemodel";
	std::string default_config_file    = "/home/jinbeibei/mv3d/examples/FRCNN/config/didi_config.json";

  
  

	
	API::Set_Config(default_config_file);

	
	caffe::Timer load;
	load.Start();
	FRCNN_API::Detector Detectorcar(proto_file, model_file);
	LOG(INFO) << "load:-->" << load.MicroSeconds()/1000 <<" ms";
	load.Stop();




	for(int i = 0; i < 5; i++){
		//for(int i = 0; i < bv_channels; i++){
		
		if(!bird[index_bird].data){
			
			return;
		}
			
		bv_image.push_back(bird[index_bird]);
	
		index_bird=index_bird+1;
	
	}
	
	
	if(caffe::Frcnn::FrcnnParam::use_fv){
		vector<cv::Mat> fv_image;
		for(int i = 0; i < 2; i++){
			while(index_front<len_front){
			
			
				//string fv_path = fv_image_path + postfix[i];
				//cv::Mat fv_mat = front
				fv_image.push_back(front[index_front]);
				index_front=index_front+1;
				
			}
		}
		for(int i = 0; i < 2; i++){
			while(index_back<len_back){

				//string bkv_path = bkv_image_path + postfix[i];
				//cv::Mat bkv_mat = cv::imread(bkv_path, CV_LOAD_IMAGE_GRAYSCALE);
				fv_image.push_back(back[index_back]);
				index_back=index_back+1;
				}
		}
		
		Detectorcar.preprocess_pub(fv_image, 1);
		Detectorcar.preprocess_pub(fv_image,1);
		
	}

	vector<float> bv_size = caffe::Frcnn::FrcnnParam::bv_size; 
	std::vector<float> im_info(3);

	//bv_size:1000*1000
	//im_info[0] = 1000; //height
	//im_info[1] = 1000; 

	im_info[0] = bv_size[0]; //height
	im_info[1] = bv_size[1]; //width
	im_info[2] = 1;
	//LOG(INFO) << im_info[0];

	//this->preprocess(bird, 0);
	//FRCNN_API::Detector::preprocess(bv_image, 0);
	//FRCNN_API::Detector::preprocess(im_info, 1);
	ROS_INFO("cccccc");

	//Detectorcar.preprocess_pub(im_info,1);

	caffe::Timer predict_timer_preprocess_bv_image;
	predict_timer_preprocess_bv_image.Start();
	Detectorcar.preprocess_pub(bv_image,0);
	predict_timer_preprocess_bv_image.Stop();
	LOG(INFO) << "predict_timer_preprocess_bv_image:-->" << predict_timer_preprocess_bv_image.MicroSeconds()/1000 <<" ms";
	//fout<<"predict_timer_preprocess_bv_image:-->" << predict_timer_preprocess_bv_image.MicroSeconds()/1000 <<" ms"<<"\n";



	caffe::Timer predict_timer_preprocess_im_info;
	predict_timer_preprocess_im_info.Start();
	Detectorcar.preprocess_pub(im_info,1);
	predict_timer_preprocess_im_info.Stop();
	LOG(INFO) << "predict_timer_preprocess_im_info:-->" << predict_timer_preprocess_im_info.MicroSeconds()/1000 <<" ms";

	

	vector<std::string> blob_names(3);
	blob_names[0] = "rois";
	blob_names[1] = "cls_prob";
	blob_names[2] = "bbox_pred";

	caffe::Timer predict_timer;
	predict_timer.Start();
	//vector<boost::shared_ptr<caffe::Blob<float> > > output = FRCNN_API::Detector::predict(blob_names);
	vector<boost::shared_ptr<caffe::Blob<float> > > output = Detectorcar.predict_pub(blob_names);
	LOG(INFO) << "predict:-->" << predict_timer.MicroSeconds()/1000 <<" ms";
	predict_timer.Stop();



	boost::shared_ptr<caffe::Blob<float> > rois(output[0]);
	boost::shared_ptr<caffe::Blob<float> > cls_prob(output[1]);
	boost::shared_ptr<caffe::Blob<float> > bbox_pred(output[2]);
	const int box_num = bbox_pred->num();
	const int cls_num = cls_prob->channels();
	caffe::Frcnn::FrcnnParam::n_classes=2;
	//CHECK_EQ(cls_num, caffe::Frcnn::FrcnnParam::n_classes);
	score.clear();
	results_corners.clear();


	for (int cls = 1; cls < cls_num; cls++) {
		vector<caffe::Frcnn::Cubic<float> > cubics;
		vector<caffe::Frcnn::BBox<float> > bboxs;
		float max_score = 0;
		int max_index = 0;
		for (int i = 0; i < box_num; i++) {
			float score = cls_prob->cpu_data()[i * cls_num + cls];
			//LOG(INFO) << i <<"-th score:-->" << score;
			if (score < caffe::Frcnn::FrcnnParam::test_score_thresh)
				break;

			if (max_score - score < caffe::Frcnn::FrcnnParam::eps) {
				max_score = score;
				max_index = i;
			}
		}

		caffe::Frcnn::Point6f<float> roi(rois->cpu_data()[(max_index * 7) + 1],
				rois->cpu_data()[(max_index * 7) + 2], rois->cpu_data()[(max_index * 7) + 3],
				rois->cpu_data()[(max_index * 7) + 4], rois->cpu_data()[(max_index * 7) + 5],
				rois->cpu_data()[(max_index * 7) + 6]);



		
		caffe::Timer predict_timer_lidar3dtobv;
		predict_timer_lidar3dtobv.Start();
		caffe::Frcnn::Point4f<float> bv = caffe::Frcnn::lidar_3d_to_bv(roi);
		LOG(INFO) << "time_to_lidar3dtobv:-->" << predict_timer_lidar3dtobv.MicroSeconds()/1000 <<" ms";
		predict_timer_lidar3dtobv.Stop();





		vector<float> deltas;
		for (int d = 0; d < 24; d++) {
			deltas.push_back(bbox_pred->cpu_data()[(max_index * cls_num + cls) * 24 + d]);
		}
		vector<float> corners ;

		
		caffe::Timer predict_timer_bbox_transform_corners_inv;
		predict_timer_bbox_transform_corners_inv.Start();
		corners= caffe::Frcnn::bbox_transform_corners_inv(roi, deltas);
		LOG(INFO) << "predict_timer_bbox_transform_corners_inv:-->" << predict_timer_bbox_transform_corners_inv.MicroSeconds()/1000 <<" ms";
		predict_timer_bbox_transform_corners_inv.Stop();





		results_corners.push_back(corners);
		score.push_back(max_score);

	}


  //return results_corners;
}

void pcCallback(const sensor_msgs::PointCloud2ConstPtr& msg)  
{
  
  // sprintf(pc_time,"%10d%09d",(msg->header).stamp.sec,(msg->header).stamp.nsec);
  // cout<<pc_time<<endl;

  //pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;   
  //pcl_conversions::toPCL(*msg, *cloud);
  //cl=new pcl::PointCloud<pcl::PointXYZ>;
  //pcl::fromPCLPointCloud2(*cloud,*cl);
	//clock_t begin=clock();
 	//ros::Time begin=ros::Time::now();/






	ofstream fout("output.txt");


/*
	int gpu_id = 0;
	caffe::Caffe::set_mode(caffe::Caffe::GPU);		
	std::string proto_file             = "/home/jinbeibei/mv3d/models/FRCNN/mv3d/vgg16/test.proto";
	std::string model_file             = "/home/jinbeibei/mv3d/models/FRCNN/snapshot/vgg16_mv3d_bv_loss_iter_4000.caffemodel";
	std::string default_config_file    = "/home/jinbeibei/mv3d/examples/FRCNN/config/didi_config.json";

  
  

	
	API::Set_Config(default_config_file);

	
	caffe::Timer load;
	load.Start();
	FRCNN_API::Detector Detectorcar(proto_file, model_file);
	LOG(INFO) << "load:-->" << load.MicroSeconds()/1000 <<" ms";
	fout << "load-->" << load.MicroSeconds()/1000 <<" ms" << "\n";
	load.Stop();
*/



	caffe::Timer predict_timer_begin;
	caffe::Timer predict_timer_project;
	caffe::Timer predict_timer_detect;
	predict_timer_begin.Start();
	//double begin=ros::Time::now().toSec();	
	//ROS_INFO("begin_time:%ld",begin);
	//fout << "begin=ros::Time::now().toSec():" << begin << "\n";
	//double begin=ros::Time::now().toSec();
	pcl::fromROSMsg(*msg, *cl);

	//ROS_INFO("get cl");
	// pc_vis(cl);
	cv::Mat* biredviews = NULL;
	cv::Mat* frontviews = NULL;
	cv::Mat* backviews = NULL;
	
	predict_timer_project.Start();
	projectPCL(cl, biredviews, frontviews, backviews);
	predict_timer_project.Stop();
	LOG(INFO) << "project_timer_project-->" << predict_timer_project.MicroSeconds()/1000 <<" ms";
	fout << "project_timer_project-->" << predict_timer_project.MicroSeconds()/1000 <<" ms" << "\n";
	//clock_t time_to_project=clock();/

	//ros::Time time_to_project=ros::Time::now();
	//double time_to_project=ros::Time::now().toSec();
	
	//fout << "time_to_project=ros::Time::now().toSec():" << time_to_project << "\n";
	//ros::Time time_to_project=ros::Time::now();
	
	
	//mv_vis(biredviews, frontviews, backviews);
	  
	std::vector<std::vector<float> > results_corners_car;
	//std::vector<float> &results_cornerscar;
	std::vector<float> score_car;

	predict_timer_detect.Start();
	detectCar(biredviews, frontviews, backviews,results_corners_car,score_car);
	predict_timer_detect.Stop();

	LOG(INFO) << "predict_timer_detect:-->" << predict_timer_detect.MicroSeconds()/1000 <<" ms";
	fout << "predict_timer_detect:" << predict_timer_detect.MicroSeconds()/1000 <<" ms" << "\n";
	

	predict_timer_begin.Stop();
	LOG(INFO) << "timer_to_begin:-->" << predict_timer_begin.MicroSeconds()/1000 <<" ms";
	fout << "timer_to_begin:" << predict_timer_begin.MicroSeconds()/1000 <<" ms" << "\n";
	

	

	fout << flush; fout.close();
	//clock_t time_to_detect=clock();
	//ros::Time time_to_detect=ros::Time::now();

	//double time_to_detect=ros::Time::now().toSec();
       
	
	//double secs_1=(time_to_project-begin)/CLOCKS_PER_SEC;
	//double secs_2=(time_to_detect-begin)/CLOCKS_PER_SEC;

	//double secs_1=(time_to_project-begin);
	//double secs_2=(time_to_detect-begin);
	//ROS_INFO("begin_time:%ld",begin);
	//ROS_INFO("time to project:%ld",secs_1);
	//ROS_INFO("time to detect:%ld",secs_2);
	//cout<< "time to projectPCL is: "<<secs_1<<"s"<<endl;
	//cout<< "time to detectcar is: "<<secs_2<<"s"<<endl;
	//exit(0);
}

int main(int argc, char **argv)  
{  
  // cl = new pcl::PointCloud<pcl::PointXYZ>;
  ros::init(argc, argv, "detect_car");  
  ros::NodeHandle nh_;  
  ROS_INFO("111111");
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: demo_frcnn_api <args>\n\n"
      "args:\n"
      "  --gpu          0       use 7-th gpu device, default is cpu model\n"
      "  --model        file    protocol buffer text file\n"
      "  --weights      file    Trained Model\n"
      "  --default_c    file    Default Config File\n"
      "  --image_list   file    input image list\n"
      "  --image_root   file    input image dir\n"
      "  --max_per_image   file limit to max_per_image detections\n"
      "  --out_file     file    output amswer file");

  // Run tool or show usage.
 
  caffe::GlobalInit(&argc, &argv);
  
  ROS_INFO("mmmnnn");
 // API::Detector detectorcar(proto_file, model_file);
  ROS_INFO("nnnnnn");
  
  // ros::param::get("~format1",format1);
  //int i = 0;
  ros::Subscriber pc_sub = nh_.subscribe("/velodyne_points", 10000, pcCallback);
  ROS_INFO("222222");
  ros::spin();  
  return 0; 
}
