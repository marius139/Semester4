//General libraries
#include <string>
#include <iostream>
#include <time.h>
#include <math.h>
#include <vector>
#include <cstring>
#include <curses.h>
#include <cstddef>
#include <cstdlib>
#include <cmath>

//OpenCV and Intel Realsense libraries
#include <librealsense2/rs.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/imgproc/imgproc.hpp"

//JACO librabies
#include <dlfcn.h>
#include "Lib_Examples/Kinova.API.CommLayerUbuntu.h"
#include "Lib_Examples/KinovaTypes.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

using namespace rs2;
using namespace std;
using namespace cv;

const double PI = 3.14159265358979323846;
const int CLOSE_FINGER = 6100;

//Gripper open or closed
int gripperValue = 0;

Point2f vertices[4];

Point2f cent;
Point2f centSelect;

float dist_to_point;
float dist_to_point1;
float dist_to_point2;
float dist_to_pointSelect;

//OpenCV images
Mat grayimage;
Mat imgCanny;
Mat imgContours;
Mat blurImg;
Mat imgDilate; 

//OpenCV variables
Mat element = getStructuringElement( MORPH_RECT, Size (7, 7) );
const int lowthresh = 9;
vector<vector<Point>> contours;
vector<vector<Point>> contours1;
Scalar BGR = Scalar(255, 0, 0);

//Set up JACO SDK 
CartesianPosition currentCommand;
AngularPosition currentCommandAng;
//Handle for the library's command layer.
void * commandLayer_handle;

//Function pointers to the functions we need
int (*MyInitAPI)();
int (*MyCloseAPI)();
int (*MySendBasicTrajectory)(TrajectoryPoint command);
int (*MyGetDevices)(KinovaDevice devices[MAX_KINOVA_DEVICE], int &result);
int (*MySetActiveDevice)(KinovaDevice device);
int (*MyMoveHome)();
int (*MyInitFingers)();
int (*MyGetQuickStatus)(QuickStatus &);
int (*MyGetCartesianCommand)(CartesianPosition &);
int (*MyGetAngularCommand)(AngularPosition &);
int (*MyStartControlAPI)();
int (*MyEraseAllTrajectories)();

//Pipeline variables for realsense camera
rs2::colorizer c;// Helper to colorize depth images
rs2::pipeline piper;
rs2::config cfg;
rs2::align align_to_depth(RS2_STREAM_DEPTH);
rs2::align align_to_color(RS2_STREAM_COLOR);

//Obect selected
int objectSelect = 0;

//Function used for detecting an object of interest, and extracting distance and position of center pixel
vector <float> objectDetection()
{	
	rs2::frameset frameset;

	// Using the align object, we block the application until a frameset is available
	frameset = piper.wait_for_frames();

	//Get color image from realsense
	frameset = align_to_color.process(frameset);
	frame color = frameset.get_color_frame();
	Mat image(Size(640, 360), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

	//Get depth frame from realsense
	frameset = align_to_depth.process(frameset);
	auto depth = frameset.get_depth_frame();
	auto colorized_depth = c.colorize(depth);
	Mat ImageDepth(Size(640, 360), CV_8UC3, (void*)colorized_depth.get_data(), Mat::AUTO_STEP);
	Mat depthThresh1 = Mat(ImageDepth.rows, ImageDepth.cols, CV_8UC3);
		

	//Show image
	cv::imshow("ColorImg heeri", image);
	//Convert to grayscale
	cvtColor(image, grayimage, COLOR_BGR2GRAY);

	//Blur image
	blur(grayimage, blurImg, Size(5, 5));
	//Apply canny edge detection
	Canny(blurImg, imgDilate, lowthresh, lowthresh * 3, 3);

	//Dilate to connect edges
	dilate(imgDilate, imgCanny, element);		

	//Find contours and draw
	findContours(imgCanny, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(imgCanny, contours, -1, 255, -1);

	//Keep all parts of the image within a certain distance range from the camera
	for (int x = 0; x < imgCanny.cols; x++) {
		for (int y = 0; y < imgCanny.rows; y++) {
			dist_to_point = depth.get_distance(x, y);
			dist_to_point1 = depth.get_distance(x + 1, y + 1);

			
			if (dist_to_point <= 0.50 && dist_to_point >= 0.20 && dist_to_point != 0 || 
			    dist_to_point1 <= 0.50 && dist_to_point1 >= 0.20 && dist_to_point1 != 0){
				imgCanny.at<uchar>(y, x) = imgCanny.at<uchar>(y, x);
			} else {
				imgCanny.at<uchar>(y, x) = 0;
			}
		}
	}

	contours1.clear();
	contours.clear();

	//Find new contours 
	findContours(imgCanny, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	Mat imgCanny1 = Mat(imgCanny.rows,imgCanny.cols, CV_8UC1);

	if (!contours.empty()) {
		for (int x = 0; x < contours.size(); x++) {
			double area = contourArea(contours[x], 0);
			
			//dont consider very small contours 
			if (area > 500) {

				//Find circle and rectangle for contour
				minEnclosingCircle(contours[x], cent, rad);
				RotatedRect res = minAreaRect(contours[x]);
				res.points(vertices);

				//get distance to center pixel of minEnclosing circle
				dist_to_point2 = depth.get_distance(cent.x, cent.y);
				cout << "dist to object: " << dist_to_point2 << endl;

				//Calculate widt of bottle
				double WidthRect = sqrt(pow((vertices[0].x - vertices[1].x), 2) + pow((vertices[0].y - vertices[1].y), 2));
				double HeightRect = sqrt(pow((vertices[1].x - vertices[2].x), 2) + pow((vertices[1].y - vertices[2].y), 2));
				double widthRatio;
				if (WidthRect > HeightRect) {
					widthRatio = HeightRect / 640;
				}
				else {
					widthRatio = WidthRect / 640;
				}
				double FOVarea = tan(0.606)*dist_to_point2 * 2;
				double objectWidth = FOVarea * widthRatio;

				//Save contours in vector if it is able to be picked up(is between the desired width)
				if (objectWidth > 0.04 && objectWidth < 0.12) {
					cout << "widt: " << x << " " << objectWidth << endl;
					drawContours(imgCanny1, contours, x, 255, -1);
					contours1.push_back(contours[x]);
				}
			}
		}
	}

	float rad;
	if (!contours1.empty()) {
	
		for (int x = 0; x < contours1.size(); x++) {
			double area = contourArea(contours1[x], 0);

			
			RotatedRect re = minAreaRect(contours1[x]);
			//A coordinate vector, for all the lines of the rectangle
			re.points(vertices); 

			//If the current contour is the selected one
			if(x == objectSelect){
			minEnclosingCircle(contours1[x], centSelect, rad);
			dist_to_pointSelect = depth.get_distance((int)cent.x, (int)cent.y);

				//This for loop draws the rectangle, using lines. (selected=green, others = red)
				for (int i = 0; i < 4; i++) {
					line(imgCanny1, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
					line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
				}
			}else{

				for (int i = 0; i < 4; i++) {
					line(imgCanny1, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
					line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
				}
			}
		
		}
	} 
		//Show image with pickupable ubjects marked
		cv::imshow("ColorImg heeri", image);
		cv::waitKey(10);
	
	
	//Return distance to object and center pixel
	vector<float> objectPosition;
	objectPosition.clear();
	objectPosition.push_back(centSelect.x);
	objectPosition.push_back(centSelect.y); 
	objectPosition.push_back(dist_to_pointSelect);

	return objectPosition;
}

//Calculate position of object in relation to robot base
vector <float> calculatePosition(vector<float> objectPosition)
{
	vector<float> objectCoordinates;

	if (objectPosition.size() != 0){
		double PI = 3.14159265;
		float cAng = 15; //Camera tilt relative to world plane in degrees, maybe -15??
		float xFOV = 69.4f;
		float yFOV = 42.5f;

		float xRatio =  xFOV/640; //img.rows
		float yRatio = yFOV/360; //img.cols

		float r = objectPosition[2];
	
		float xPos = objectPosition[0] - 320;	
		float yPos = objectPosition[1] - 180;	

		float theta = xPos * xRatio;
		float phi = yPos * yRatio;

		float martin = cos(70*PI / 180)*r;
		
		float x = r * sin(theta * PI / 180) * cos(phi * PI / 180); //x
		float y = r * sin(theta * PI / 180) * sin(phi * PI / 180)-0.0845+martin; //y
		float z = r * cos(theta * PI / 180)-0.109+0.03;//z

		//Camera to object transformation
		float Cam2Obj[4][4] = { {1,0,0,x},{0,1,0,y},{0,0,1,z},{0,0,0,1}};
		float Ee2Cam[4][4] = { {-0.2588,0,0.9659,0.109},{0,1,0,0.035},{-0.9659,0,-0.2588, 0.0777},{0,0,0,1}};
		
		//Get robot position
		(*MyGetCartesianCommand)(currentCommand);
		float Xee = currentCommand.Coordinates.X;
		float Yee = currentCommand.Coordinates.Y;
		float Zee = currentCommand.Coordinates.Z;
		float ThetaXee = currentCommand.Coordinates.ThetaX;
		float ThetaYee = currentCommand.Coordinates.ThetaY;
		float ThetaZee = currentCommand.Coordinates.ThetaZ;

		
		float B2Obj [4][4];		
		float B2Cam [4][4];

		//Base to endefector transformation
		float B2Ee[4][4] = {{cos(ThetaYee)*cos(ThetaZee),-cos(ThetaYee)*sin(ThetaZee),sin(ThetaYee),Xee},{cos(ThetaXee)*sin(ThetaZee)+cos(ThetaZee)*sin(ThetaXee)*sin(ThetaYee),cos(ThetaXee)*cos(ThetaZee)-sin(ThetaXee)*sin(ThetaYee)*sin(ThetaZee),-cos(ThetaYee)*sin(ThetaXee),Yee},{sin(ThetaXee)*sin(ThetaZee)-cos(ThetaXee)*cos(ThetaZee)*sin(ThetaYee),cos(ThetaZee)*sin(ThetaXee)+cos(ThetaXee)*sin(ThetaYee)*sin(ThetaZee),cos(ThetaXee)*cos(ThetaYee),Zee},{0,0,0,1},};

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				B2Cam[i][j] = 0;
			}
		}

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				B2Obj[i][j] = 0;
			}
		}


		// Multiplying matrix a and b and storing in array mult.
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 4; ++k) {
					B2Cam[i][j] += B2Ee[i][k] * Ee2Cam[k][j];
				}
			}
		}
	
		// Multiplying matrix a and b and storing in array mult.
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 4; ++k) {
					B2Obj[i][j] += B2Ee[i][k] * Cam2Obj[k][j];
				}
			}
		}

		//Save coordinates of object
		objectCoordinates.clear();

		objectCoordinates.push_back(B2Obj[0][3]);
		objectCoordinates.push_back(B2Obj[1][3]);
		objectCoordinates.push_back(B2Obj[2][3]);


		
	} else {
		objectCoordinates.clear();
	}

	

	return objectCoordinates;
}

//Function for handeling all movement
void manualMove(int buttonValue, vector<float> objectPosition) {
	vector<float> objectCoordinates;
	int result;
	
	void* commandLayer_handle;
        //Function pointers to the functions we need
        int (*MyInitAPI)();
        int (*MyCloseAPI)();
        int (*MySendBasicTrajectory)(TrajectoryPoint command);
        int (*MyStartControlAPI)();
        int (*MyEraseAllTrajectories)();
	int (*MyGetAngularCommand)(AngularPosition &);

        //We load the library (Under Windows, use the function LoadLibrary)
        commandLayer_handle = dlopen("Kinova.API.USBCommandLayerUbuntu.so",RTLD_NOW|RTLD_GLOBAL);

        //We load the functions from the library (Under Windows, use GetProcAddress)
        MyInitAPI = (int (*)()) dlsym(commandLayer_handle,"InitAPI");
        MyCloseAPI = (int (*)()) dlsym(commandLayer_handle,"CloseAPI");
        MySendBasicTrajectory = (int (*)(TrajectoryPoint)) dlsym(commandLayer_handle,"SendBasicTrajectory");
        MyStartControlAPI = (int (*)()) dlsym(commandLayer_handle,"StartControlAPI");
        MyEraseAllTrajectories = (int (*)()) dlsym(commandLayer_handle,"EraseAllTrajectories");
	MyGetAngularCommand = (int (*)(AngularPosition &)) dlsym(commandLayer_handle,"GetAngularCommand");
	result = (*MyEraseAllTrajectories)();

	TrajectoryPoint pointToSend;	
	pointToSend.InitStruct();
	TrajectoryPoint pointToSendAng;	
	pointToSendAng.InitStruct();
	float cartesianPose[6];
	float xGoal;
	float yGoal;
	float zGoal;
	int reached = 0;
	
	switch (buttonValue) {
	case 'i': //Open/close gripper
		pointToSend.Position.Type = CARTESIAN_POSITION;
		pointToSend.Position.HandMode = POSITION_MODE;
		
			cout << "Use Gripper acquired " << endl;

			if (gripperValue == 0) {
				//We get the actual angular command of the robot.
				(*MyGetCartesianCommand)(currentCommand);
				usleep(1000);

				//Close gripper
				pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X;
				pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
				pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
				pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
				pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
				pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

				pointToSend.Position.Fingers.Finger1 = CLOSE_FINGER;
				pointToSend.Position.Fingers.Finger2 = CLOSE_FINGER;


				//Send this command to the JACO
				MySendBasicTrajectory(pointToSend);
				usleep(1000);

				gripperValue = 1;
				cout << "Close grip" << endl;
				break;
			}else {
				//We get the actual angular command of the robot.
				(*MyGetCartesianCommand)(currentCommand);

				//Open gripper
				pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X;
				pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
				pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
				pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
				pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
				pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

				pointToSend.Position.Fingers.Finger1 = 0;
				pointToSend.Position.Fingers.Finger2 = 0;

				//Send this command to the JACO
				(*MySendBasicTrajectory)(pointToSend);
				usleep(1000);

				gripperValue = 0;
				cout << "Open grip" << endl;
				break;
			}
		break;

	case 'c':  //Upward
		
			cout << "Moving the robot up" << endl;

			//We get the actual angular command of the robot.
			(*MyGetCartesianCommand)(currentCommand);
			pointToSend.Position.Type = CARTESIAN_POSITION;
			pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			//Determine which direction the EE should move in cartesian space
			pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X;
			pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
			pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z + 0.02f;
			pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
			pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
			pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

			//Send this command to the JACO
			(*MySendBasicTrajectory)(pointToSend);
			usleep(100);
	         break;

	case 'f':  //Downward
		
			cout << "Moving the robot down" << endl;
			
			//We get the actual angular command of the robot.
			(*MyGetCartesianCommand)(currentCommand);
			pointToSend.Position.Type = CARTESIAN_POSITION;
			pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			//Determine which direction the EE should move in cartesian space
			pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X;
			pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
			pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z - 0.02f;
			pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
			pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
			pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

			//Send this command to the JACO
			(*MySendBasicTrajectory)(pointToSend);
			usleep(100);
		 break;
		
	case 'a':  //Cycle to next object
		
			cout << "Select next object" << endl;
			objectSelect++;
			usleep(1000);

			if(objectSelect>contours1.size()-1){
			objectSelect = 0;
			}
			
			usleep(10000);
		 break;

	
	case KEY_UP: //Go forwards
			
			cout << "Moving the robot forwards" << endl;
			pointToSend.Position.Type = CARTESIAN_POSITION;
			pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			
			(*MyGetCartesianCommand)(currentCommand);

			//Determine which direction the EE should move in cartesian space
			pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X + 0.02f;
			pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
			pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
			pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
			pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
			pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

			//Send this command to the JACO
			(*MySendBasicTrajectory)(pointToSend);
			usleep(100);
		 break;

	
	case KEY_DOWN:  //Go backwards
			
			cout << "Moving the robot backwards" << endl;

			//We get the actual angular command of the robot.
			(*MyGetCartesianCommand)(currentCommand);
			pointToSend.Position.Type = CARTESIAN_POSITION;
			pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			//Determine which direction the EE should move in cartesian space
			pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X - 0.02f;
			pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
			pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
			pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
			pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
			pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

			//Send this command to the JACO
			(*MySendBasicTrajectory)(pointToSend);
			usleep(100);
		 break;

	
	case KEY_RIGHT:  //Move left
			
			cout << "Moving the robot to the Left" << endl;

			//We get the actual angular command of the robot.
			(*MyGetCartesianCommand)(currentCommand);
			pointToSend.Position.Type = CARTESIAN_POSITION;
			pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			//Determine which direction the EE should move in cartesian space
			pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X;
			pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y + 0.02f;
			pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
			pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
			pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
			pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

			//Send this command to the JACO
			(*MySendBasicTrajectory)(pointToSend);
			usleep(100); //500000
		 break;
	
	
	case KEY_LEFT:  //Move right
			
			cout << "Moving the robot to the Right" << endl;

			//We get the actual angular command of the robot.
			(*MyGetCartesianCommand)(currentCommand);
			pointToSend.Position.Type = CARTESIAN_POSITION;
			pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			//Determine which direction the EE should move in cartesian space
			pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X;
			pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y - 0.02f;
			pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
			pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
			pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
			pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

			//Send this command to the JACO
			(*MySendBasicTrajectory)(pointToSend);
			usleep(100);
		 break;

	
	case 'b':  //go to the starting position 
			//pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			pointToSend.Position.Type = CARTESIAN_POSITION;
			cout << "Moving" << endl;
			
			//We get the actual angular command of the robot.
			(*MyGetCartesianCommand)(currentCommand);
			pointToSend.Position.HandMode = HAND_NOMOVEMENT;
			pointToSend.Position.CartesianPosition.X = 0.469788f;
			pointToSend.Position.CartesianPosition.Y = 0.264659f;
			pointToSend.Position.CartesianPosition.Z = 0.502227f;
			pointToSend.Position.CartesianPosition.ThetaX = -1.80912f;
			pointToSend.Position.CartesianPosition.ThetaY = 1.235f;
			pointToSend.Position.CartesianPosition.ThetaZ =	0.240239f;
	
			(*MySendBasicTrajectory)(pointToSend);
			usleep(5000000);

			pointToSend.Position.CartesianPosition.X = -0.0903951f;
			pointToSend.Position.CartesianPosition.Y = 0.452799f;
			pointToSend.Position.CartesianPosition.Z = 0.446799f;
			pointToSend.Position.CartesianPosition.ThetaX = -1.80912f;
			pointToSend.Position.CartesianPosition.ThetaY = 1.235f;
			pointToSend.Position.CartesianPosition.ThetaZ =	0.240239f;
	
			(*MySendBasicTrajectory)(pointToSend);
			usleep(1000000);
		 break;

	
	case 'j': //Automatic grip
			
			cout << "Move to object, and grab it" << endl;	
			objectCoordinates = calculatePosition(objectPosition);
			(*MyGetCartesianCommand)(currentCommand);
			pointToSend.Position.Type = CARTESIAN_POSITION;
			if (objectCoordinates.size() != 0) {

				(*MyGetCartesianCommand)(currentCommand);
			
				xGoal = objectCoordinates[0];
				yGoal = objectCoordinates[1];
				zGoal = objectCoordinates[2];
				cout << "xGoal " << xGoal << endl;
				cout << "yGoal " << yGoal << endl;
				cout << "zGoal " << zGoal << endl;
				
				//Determine which direction the EE should move in cartesian space
				pointToSend.Position.CartesianPosition.X = xGoal;
				pointToSend.Position.CartesianPosition.Y = yGoal;
				pointToSend.Position.CartesianPosition.Z = zGoal;
				pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
				pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
				pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;
				
				(*MySendBasicTrajectory)(pointToSend);
				usleep(3000000);
				
				result = (*MyEraseAllTrajectories)();
				usleep(1000000);

				(*MyGetCartesianCommand)(currentCommand);
				usleep(1000);

				//close gripper
				pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X;
				pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
				pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
				pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
				pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
				pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;

				pointToSend.Position.Fingers.Finger1 = CLOSE_FINGER;
				pointToSend.Position.Fingers.Finger2 = CLOSE_FINGER;

				MySendBasicTrajectory(pointToSend);
				cout << "Close grip" << endl;
						
				usleep(1000000);
				//move back
				pointToSend.Position.CartesianPosition.X = currentCommand.Coordinates.X-0.2f;
				pointToSend.Position.CartesianPosition.Y = currentCommand.Coordinates.Y;
				pointToSend.Position.CartesianPosition.Z = currentCommand.Coordinates.Z;
				pointToSend.Position.CartesianPosition.ThetaX = currentCommand.Coordinates.ThetaX;
				pointToSend.Position.CartesianPosition.ThetaY = currentCommand.Coordinates.ThetaY;
				pointToSend.Position.CartesianPosition.ThetaZ = currentCommand.Coordinates.ThetaZ;
				
				MySendBasicTrajectory(pointToSend);
				usleep(1000000);

				gripperValue = 1;

				pointToSend.Position.Type = CARTESIAN_POSITION;
				cout << "Moving" << endl;
			
				//Move back to starting position
				pointToSend.Position.CartesianPosition.X = 0.469788f;
				pointToSend.Position.CartesianPosition.Y = 0.264659f;
				pointToSend.Position.CartesianPosition.Z = 0.502227f;
				pointToSend.Position.CartesianPosition.ThetaX = -1.80912f;
				pointToSend.Position.CartesianPosition.ThetaY = 1.235f;
				pointToSend.Position.CartesianPosition.ThetaZ =	0.240239f;
	
				(*MySendBasicTrajectory)(pointToSend);
				usleep(3000000);

												
				pointToSend.Position.CartesianPosition.X = -0.0903951f;
				pointToSend.Position.CartesianPosition.Y = 0.452799f;
				pointToSend.Position.CartesianPosition.Z = 0.446799f;
				pointToSend.Position.CartesianPosition.ThetaX = -1.80912f;
				pointToSend.Position.CartesianPosition.ThetaY = 1.235f;
				pointToSend.Position.CartesianPosition.ThetaZ =	0.240239f;
	
				(*MySendBasicTrajectory)(pointToSend);
				usleep(1000000);
			
			
			} else {
				cout << "No object detected. Launch computer vision or move robot to different position." << endl;
				usleep(2000000);
			}


			 break;


	case 'r':  	//Close program
			dlclose(commandLayer_handle);
			cout << endl << "C L O S I N G   A P I" << endl;
			result = (*MyCloseAPI)();
			usleep(500000);
			system("killall gnome-terminal-server");
		 break;

	default: 
		 cout<<"Input recorded but not recognized" << endl;
		 usleep(2000000);
		 break;
	}
}

//Function used for interpreting commands send from the Itongue
void userInput(vector<float> objectPosition)
{	
	initscr();
	cbreak();
	noecho();
	keypad(stdscr,TRUE);
	nodelay(stdscr,TRUE);
	
	int ch = getch();
	if (ch == ERR) {
		//cout<<"Waiting for an input"<<endl;
	} else {
		manualMove(ch, objectPosition);
		flushinp();
	} 
}


int main(int argc, char* argv[])
{	
	//Configure realsense image stream
	cfg.enable_stream(RS2_STREAM_DEPTH, -1, 640, 360);
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8, 30); //30	
	piper.start(cfg);
	waitKey(3000);

	//We load the library
	commandLayer_handle = dlopen("Kinova.API.USBCommandLayerUbuntu.so",RTLD_NOW|RTLD_GLOBAL);

	//We load the functions from the library (Under Windows, use GetProcAddress)
	MyInitAPI = (int (*)()) dlsym(commandLayer_handle,"InitAPI");
	MyCloseAPI = (int (*)()) dlsym(commandLayer_handle,"CloseAPI");
	MyMoveHome = (int (*)()) dlsym(commandLayer_handle,"MoveHome");
	MyInitFingers = (int (*)()) dlsym(commandLayer_handle,"InitFingers");
	MyGetDevices = (int (*)(KinovaDevice devices[MAX_KINOVA_DEVICE], int &result)) dlsym(commandLayer_handle,"GetDevices");
	MySetActiveDevice = (int (*)(KinovaDevice devices)) dlsym(commandLayer_handle,"SetActiveDevice");
	MySendBasicTrajectory = (int (*)(TrajectoryPoint)) dlsym(commandLayer_handle,"SendBasicTrajectory");
	MyGetQuickStatus = (int (*)(QuickStatus &)) dlsym(commandLayer_handle,"GetQuickStatus");
	MyGetCartesianCommand = (int (*)(CartesianPosition &)) dlsym(commandLayer_handle,"GetCartesianCommand");
	MyStartControlAPI = (int (*)()) dlsym(commandLayer_handle,"StartControlAPI");
        MyEraseAllTrajectories = (int (*)()) dlsym(commandLayer_handle,"EraseAllTrajectories");


	//Verify that all functions has been loaded correctly
	if ((MyInitAPI == NULL) || (MyCloseAPI == NULL) || (MySendBasicTrajectory == NULL) || (MyGetDevices == NULL) || 
           (MyStartControlAPI == NULL) || (MyEraseAllTrajectories == NULL) || (MySetActiveDevice == NULL) || 
	   (MyGetCartesianCommand == NULL) || (MyMoveHome == NULL) || (MyInitFingers == NULL)) {
		cout << "* * *  E R R O R   D U R I N G   I N I T I A L I Z A T I O N  * * *" << endl;
		return 0;
	}

	//Start jaco api
	cout << "I N I T I A L I Z A T I O N   C O M P L E T E D" << endl << endl;
	int result = (*MyInitAPI)();
	cout << "Initialization's result :" << result << endl;
	KinovaDevice list[MAX_KINOVA_DEVICE];
	int devicesCount = (*MyGetDevices)(list, result);
	
	//Move home and initialize fingers
	(*MyMoveHome)();
	(*MyInitFingers)();

	TrajectoryPoint pointToSend;	
	pointToSend.InitStruct();
	pointToSend.Position.HandMode = HAND_NOMOVEMENT;
	pointToSend.Position.Type = CARTESIAN_POSITION;
	
	//move to starting position
	pointToSend.Position.CartesianPosition.X = 0.212321f;
	pointToSend.Position.CartesianPosition.Y = -0.257223f;
	pointToSend.Position.CartesianPosition.Z = 0.509706f;
	pointToSend.Position.CartesianPosition.ThetaX = 4.60451f;
	pointToSend.Position.CartesianPosition.ThetaY = 1.11313f;
	pointToSend.Position.CartesianPosition.ThetaZ =	0.134582f;

	pointToSend.Position.Fingers.Finger1 = CLOSE_FINGER;
	pointToSend.Position.Fingers.Finger2 = CLOSE_FINGER;
	
	(*MySendBasicTrajectory)(pointToSend);
	usleep(1000000);
	
	while(1){
		userInput( objectDetection() );
	}

	return 0;
}
