#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
 
using namespace cv;
using namespace std;

#define BOARD_GRID_L 20.7  //chessboard grid length: mm
#define BOARD_GRID_W 20.7  //chessboard grid width: mm
#define CORN_ROW  9  //corners of each row
#define CORN_COL  6  //corners of each column

int main() 
{
	ifstream fin("calib_filepath.txt"); /* 标定所用图像文件的路径 */
	ofstream fout("calib_result.txt");  /* 保存标定结果的文件 */	
	//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	cout<<"开始提取角点\n";
	int image_count=0;  /* 图像数量 */
	Size image_size;  /* 图像的尺寸 */
	Size board_size = Size(CORN_ROW,CORN_COL); /*标定板上每行、列的角点数 */
	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
	string filename;
	while (getline(fin,filename))
	{
		image_count++;		
		// 用于观察检验输出
		//cout<<"image_count = "<<image_count<<endl;		
		/* 输出检验*/
		Mat imageInput=imread(filename, IMREAD_COLOR);
		if (image_count == 1)  //读入第一张图片时获取图像宽高信息
		{
			image_size.width = imageInput.cols;
			image_size.height =imageInput.rows;			
			cout<<"image_size.width = "<<image_size.width<<endl;
			cout<<"image_size.height = "<<image_size.height<<endl;
		}
 
		/* 提取角点 */
		if (0 == findChessboardCorners(imageInput,board_size,image_points_buf))
		{			
			cout<<"can not find chessboard corners!\n"; //找不到角点
			exit(1);
		} 
		else 
		{
			Mat view_gray;
			cvtColor(imageInput,view_gray,COLOR_RGB2GRAY);
			/* 亚像素精确化 */
			find4QuadCornerSubpix(view_gray,image_points_buf,Size(5,5)); //对粗提取的角点进行精确化
			//cornerSubPix(view_gray,image_points_buf,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点
			/* 在图像上显示角点位置 */
			drawChessboardCorners(view_gray,board_size,image_points_buf,false); //用于在图片中标记角点
			//imshow("Camera Calibration",view_gray);//显示图片
			//waitKey(500);//暂停0.5S		
		}
	}
	int total = image_points_seq.size();
	cout<<"total = "<<total<<endl;
	int CornerNum=board_size.width*board_size.height;  //每张图片上总的角点数
	int cor_j;
	/*
	cout<<"开始提取角点坐标\n";
	for (int ii=0 ; ii<total ;ii++)
	{
		fout<<"第"<<ii+1<<"张图片的角点坐标: "<<endl;
		for (int cor_j=0; cor_j<CornerNum; cor_j++)
		{
			//输出所有的角点
			fout<<fixed<<setprecision(1)<<image_points_seq[ii][cor_j].x;
			fout<<", "<<fixed<<setprecision(1)<<image_points_seq[ii][cor_j].y;
			if (0 == (cor_j+1)%6) // 格式化输出，便于控制台查看
			{	
				fout<<endl;
			}
			else
			{
				fout<<"\t";
			}
		}
		fout<<endl;
	}	
	//cout<<"角点提取完成！\n";
 	*/
	//以下是摄像机标定
	cout<<"开始标定\n";
	/*棋盘三维信息*/
	Size square_size = Size(BOARD_GRID_L,BOARD_GRID_W);  /*棋盘格子大小（mm） */
	vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
	/*内外参数*/
	Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); /* 摄像机内参数矩阵 */
	vector<int> point_counts;  // 每幅图像中角点的数量
	Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
	vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
	/* 初始化标定板上角点的三维坐标 */
	int i,j,t;
	for (t=0;t<image_count;t++) 
	{
		vector<Point3f> tempPointSet;
		for (i=0;i<board_size.height;i++) 
		{
			for (j=0;j<board_size.width;j++) 
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = i*square_size.width;
				realPoint.y = j*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i=0;i<image_count;i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}	
	/* 开始标定 */
	double err_first = calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs,rvecsMat,tvecsMat,0);
	//cout<<"标定完成！\n";
	//对标定结果进行评价
	cout<<"开始评价标定结果\n";
	fout<<"重投影误差：" << err_first << "像素" << endl;
	cout<<"重投影误差：" << err_first << "像素" << endl;
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
	for (i=0;i<image_count;i++)
	{
		vector<Point3f> tempPointSet=object_points[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet,rvecsMat[i],tvecsMat[i],cameraMatrix,distCoeffs,image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
		Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
		for (int j = 0 ; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err/=  point_counts[i];   
		cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
		fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
	}   
	cout<<"所有图像总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;   
	fout<<"所有图像总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;   
	Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fout<<"相机内参数矩阵："<<endl;   
	cout<<"相机内参数矩阵："<<endl;   
	fout<<cameraMatrix<<endl<<endl;   
	cout<<cameraMatrix<<endl<<endl;   
	fout<<"畸变系数(k1,k2,p1,p2,k3)：\n";   
	cout<<"畸变系数(k1,k2,p1,p2,k3)：\n";   
	fout<<distCoeffs<<endl<<endl;   
	cout<<distCoeffs<<endl<<endl;   
	for (int i=0; i<image_count; i++) 
	{ 
		fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;   
		fout<<rvecsMat[i]<<endl;    
		/* 将旋转向量转换为相对应的旋转矩阵 */   
		Rodrigues(rvecsMat[i],rotation_matrix);   
		fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;   
		fout<<rotation_matrix<<endl;   
		fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;   
		fout<<tvecsMat[i]<<endl<<endl;   
	}   
	fout<<endl;
	/************************************************************************  
	显示定标结果  
	*************************************************************************/
	/*
	Mat mapx = Mat(image_size,CV_32FC1);
	Mat mapy = Mat(image_size,CV_32FC1);
	Mat R = Mat::eye(3,3,CV_32F);
	cout<<"保存矫正图像"<<endl;
	image_count = 1;
	fin.clear();
	fin.seekg(0,ios::beg);  //return to file beginning
	while (getline(fin,filename))
	{
		//cout<<"Frame #"<<image_count++<<"..."<<endl;
		Mat imageSource = imread(filename);
		Mat newimage = imageSource.clone();
		
		//方法一：使用initUndistortRectifyMap和remap两个函数配合实现
	initUndistortRectifyMap(cameraMatrix,distCoeffs,R,cameraMatrix,image_size,CV_32FC1,mapx,mapy);		
		remap(imageSource,newimage,mapx, mapy, INTER_LINEAR);		
		//方法二：不需要转换矩阵的方式，使用undistort函数实现
		//undistort(imageSource,newimage,cameraMatrix,distCoeffs);

		filename += "_d.jpg";
		imwrite(filename,newimage);
	}*/


	float p1 = (float)distCoeffs.at<double>(0,2);
	float p2 = (float)distCoeffs.at<double>(0,3);
	float fx = (float)cameraMatrix.at<double>(0,0);
	float fy = (float)cameraMatrix.at<double>(1,1);
	float u0 = (float)cameraMatrix.at<double>(0,2);
	float v0 = (float)cameraMatrix.at<double>(1,2);
	//内参矩阵求逆。从畸变图到矫正图用内参矩阵，反之则用逆矩阵
	Mat_<double> iR = cameraMatrix.inv(DECOMP_LU);
	float fx1 = (float)iR.at<double>(0,0);
	float fy1 = (float)iR.at<double>(1,1);
	float u01 = (float)iR.at<double>(0,2);
	float v01 = (float)iR.at<double>(1,2);
	float efl = (fx+fy)/2.0; //EFL, unit:pixls
        float pi = 3.1415926;	
	float ru, xu, yu, theta; //undistorted pixl.
	float r2;
	float xd, yd; //distorted pixl
	float a1,a2,a3; //angle
	float r_sqr_max,r_sqr_min;  //rd^2 = xd*xd+yd*yd
	float temp = 0;
	cout<<"主点偏移：u0 = "<<u0<<"像素， v0 = "<<v0<<"像素"<<endl;
	ru = 100; //实际上ru取值的变化对lens tilt的计算结果影响不大
	cout<<"ru: "<<ru<<"像素\t"<<"f: "<<efl<<"像素"<<endl;
	a1 = atan(efl/ru); 
	cout<<"以ru为半径，f为高的成像圆锥，底角 a1 = "<<a1*180/pi<<"度"<<endl;
	r_sqr_max=0;
	r_sqr_min=999999;
	/*镜头光心倾斜后，成像圆锥在成像面上的投影环不再是圆形。遍历投影环
	边缘所有点, 离光轴点(u0,v0)最远的点到光轴点的连线，即为光轴倾斜的
	方向*/
	for (theta=0; theta<2*pi; theta+=0.001)
	{
	        xu = ru*cos(theta)+u0; 
		xu = xu*fx1+u01;
		yu = ru*sin(theta)+v0; 
		yu = yu*fy1+v01;
		r2 = xu*xu+yu*yu;
		//光轴倾斜会引入切向畸变，不会引入径向畸变
		xd = xu+2*p1*xu*yu+p2*(r2+2*xu*xu);
		yd = yu+2*p2*xu*yu+p1*(r2+2*yu*yu);
		xd = xd*fx+u0;
		yd = yd*fy+v0;
		//计算（xd,yd）离光轴点(u0,v0)的距离
		xd -= u0;
		yd -= v0;
		temp = xd*xd+yd*yd; //距离的平方
		if (r_sqr_max<temp)
		{
			r_sqr_max = temp; //缓存当前最大值
		}
	       	if (r_sqr_min>temp)
		{
			r_sqr_min = temp; //缓存当前最小值
		}
	}
	float rmax = sqrt(r_sqr_max);
	float rmin = sqrt(r_sqr_min);
	cout<<"rmax: "<<rmax<<"\t"<<"rmin: "<<rmin<<endl;
	a2 = asin(ru*sin(a1)/rmax);
	a3 = a1-a2;
	a3 = 180*a3/pi; //change from radian to degree
	cout<<"光轴倾斜: "<<fixed<<setprecision(2)<<a3<<"度"<<endl;

	cout<<"结束"<<endl;	
	return 0;
}
