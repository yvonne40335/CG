#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace cv;
using namespace std;

vector<int> record;
vector<Point> pointList0, pointList1, newpointList0, newpointList1;
Point previousPoint0, previousPoint1;

void CallBackFunc0(int event, int x, int y, int flags, void* userdata)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		previousPoint0 = Point(x, y);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		Point pt(x, y);
		line((*(Mat*)userdata), previousPoint0, pt, Scalar(0, 0, 255), 2);
		pointList0.push_back(previousPoint0); //store Point of line
		pointList0.push_back(pt);
		previousPoint0 = pt;
		imshow("Image0", (*(Mat*)userdata));
		record.push_back(0);
	}
}

void CallBackFunc1(int event, int x, int y, int flags, void* userdata)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		previousPoint1 = Point(x, y);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		Point pt(x, y);
		line((*(Mat*)userdata), previousPoint1, pt, Scalar(0, 0, 255), 2);
		pointList1.push_back(previousPoint1);
		pointList1.push_back(pt);
		previousPoint1 = pt;
		imshow("Image1", (*(Mat*)userdata));
		record.push_back(1);
	}
}

void newLines(void* userdata, int flag, float t)
{
	Point newPointP, newPointQ;
	//float t = 0.5;
	Mat warp0, warp1;
	for (int i = 0; i < pointList0.size(); i += 2)
	{
		newPointP.x = (1 - t)*pointList0[i].x + t * pointList1[i].x;
		newPointP.y = (1 - t)*pointList0[i].y + t * pointList1[i].y;
		newPointQ.x = (1 - t)*pointList0[i + 1].x + t * pointList1[i + 1].x;
		newPointQ.y = (1 - t)*pointList0[i + 1].y + t * pointList1[i + 1].y;
		if (flag == 0)
		{
			//line((*(Mat*)userdata), newPointP, newPointQ, Scalar(0, 0, 255), 2);
			newpointList0.push_back(newPointP); //store Point of line
			newpointList0.push_back(newPointQ);
			//imshow("Warp0", (*(Mat*)userdata));
		}
		else
		{
			//line((*(Mat*)userdata), newPointP, newPointQ, Scalar(0, 0, 255), 2);
			newpointList1.push_back(newPointP); //store Point of line
			newpointList1.push_back(newPointQ);
			//imshow("Warp1", (*(Mat*)userdata));
		}
	}
}

cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2f pt)
{
	assert(!img.empty());
	assert(img.channels() == 3);

	int x = (int)pt.x;
	int y = (int)pt.y;

	int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
	int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
	int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
	int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

	float a = pt.x - (float)x;
	float c = pt.y - (float)y;

	uchar b = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[0] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[0] * a) * (1.f - c)
		+ (img.at<cv::Vec3b>(y1, x0)[0] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[0] * a) * c);
	uchar g = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[1] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[1] * a) * (1.f - c)
		+ (img.at<cv::Vec3b>(y1, x0)[1] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[1] * a) * c);
	uchar r = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[2] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[2] * a) * (1.f - c)
		+ (img.at<cv::Vec3b>(y1, x0)[2] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[2] * a) * c);

	return cv::Vec3b(b, g, r);
}

double test(Point X, Point P, Point Q, double u, double v)
{
	if (u < 0) {
		return sqrt(pow((X.x - P.x), 2) + pow((X.y - P.y), 2));
	}

	if (u > 1) {
		return sqrt(pow((X.x - Q.x), 2) + pow((X.y - Q.y), 2));
	}

	return std::abs(v);
}

void WarpImage(void* image, void* resultimage, int flag)
{
	Point2f psum, x, ps;
	double u, v, wsum, weight;
	double a = 0.000001, b = 1.0, c = 0.5;
	for (int i = 0; i < (*(Mat*)resultimage).cols; i++) {
		for (int j = 0; j < (*(Mat*)resultimage).rows; j++) {
			psum = Point2f(0, 0);
			wsum = 0;

			for (int k = 0; k < newpointList0.size(); k += 2)
			{
				if (flag == 0)
				{
					//利用destination的L(newLine)找出對應的p'
					//u = (x1-p1)*(q1-p1)+(x2-p2)*(q2-p2) / (Q-P)^2
					Point XP = Point(i, j) - newpointList0[k];
					Point QP = newpointList0[k + 1] - newpointList0[k];
					double L = sqrt(pow((newpointList0[k + 1].x - newpointList0[k].x), 2) + pow((newpointList0[k + 1].y - newpointList0[k].y), 2));
					u = XP.dot(QP) / pow(L, 2);

					//v = (x1-p1)*(q2-p2)+(x2-p2)*(p1-q1) / (Q-P)
					Point perQP = Point(QP.y, -QP.x);
					v = XP.dot(perQP) / L;

					//x = Ps + u*(q1-p1,q2-p2) + vdivide*(q2-p2,p1-q1)
					Point QPs = pointList0[k + 1] - pointList0[k];
					Point perQPs = Point(QPs.y, -QPs.x);
					double Ls = sqrt(pow((pointList0[k + 1].x - pointList0[k].x), 2) + pow((pointList0[k + 1].y - pointList0[k].y), 2));
					x = pointList0[k] + u * QPs + v * perQPs / Ls;

					weight = pow(pow(L, c) / (a + test(Point2f(i, j), Point2f(newpointList0[k].x, newpointList0[k].y), Point2f(newpointList0[k + 1].x, newpointList0[k + 1].y), u, v)), b);
					psum = psum + x * weight;
					wsum += weight;
				}
				else if (flag == 1)
				{
					//利用destination的L(newLine)找出對應的p'
					//u = (x1-p1)*(q1-p1)+(x2-p2)*(q2-p2) / (Q-P)^2
					Point XP = Point(i,j) - newpointList1[k]; 
					Point QP = newpointList1[k + 1] - newpointList1[k];
					double L = sqrt(pow((newpointList1[k + 1].x - newpointList1[k].x), 2) + pow((newpointList1[k + 1].y - newpointList1[k].y), 2));
					u = XP.dot(QP) / pow(L,2);

					//v = (x1-p1)*(q2-p2)+(x2-p2)*(p1-q1) / (Q-P)
					Point perQP = Point(QP.y, -QP.x);
					v = XP.dot(perQP) / L;

					//x = Ps + u*(q1-p1,q2-p2) + vdivide*(q2-p2,p1-q1)
					Point QPs = pointList1[k + 1] - pointList1[k];
					Point perQPs = Point(QPs.y, -QPs.x);
					double Ls = sqrt(pow((pointList1[k + 1].x - pointList1[k].x), 2) + pow((pointList1[k + 1].y - pointList1[k].y), 2));
					x = pointList1[k] + u * QPs + v*perQPs/Ls;

					weight = pow(pow(L, c) / (a + test(Point2f(i, j), Point2f(newpointList1[k].x, newpointList1[k].y), Point2f(newpointList1[k + 1].x, newpointList1[k + 1].y), u, v)), b);
					psum = psum + x * weight;
					wsum += weight;
				}
			}
			
			ps = psum / wsum;
			if (ps.x < 0)
				ps.x = 0;
			if (ps.y < 0)
				ps.y = 0;
			if (ps.x >= (*(Mat*)resultimage).cols)
				ps.x = (*(Mat*)resultimage).cols - 1;
			if (ps.y >= (*(Mat*)resultimage).rows)
				ps.y = (*(Mat*)resultimage).rows - 1;
			
				cv::Vec3b pixel = getColorSubpix((*(Mat*)image), ps);
				(*(Mat*)resultimage).at<cv::Vec3b>(j, i)[0] = pixel[0];
				(*(Mat*)resultimage).at<cv::Vec3b>(j, i)[1] = pixel[1];
				(*(Mat*)resultimage).at<cv::Vec3b>(j, i)[2] = pixel[2];
		}
	}
	
}

int main()
{
	Mat temp0 = imread("women.jpg");
	namedWindow("Image0", WINDOW_AUTOSIZE);
	setMouseCallback("Image0", CallBackFunc0, &temp0);
	imshow("Image0", temp0);
	Mat img0 = temp0.clone();
	

	Mat temp1 = imread("cheetah.jpg");
	namedWindow("Image1", WINDOW_AUTOSIZE);
	setMouseCallback("Image1", CallBackFunc1, &temp1);
	imshow("Image1", temp1);
	Mat img1 = temp1.clone();
	
	printf("畫線時兩邊順序要相同\n");
	printf("按a可復原線段\n");
	printf("按q即可開始進行morphing\n");

	Mat dst,frame;
	VideoWriter writer;
	int count = 0;
	int key; //按下q會開始進行warping
	key = cvWaitKey(0);
	while (true)
	{
		switch (key) {
		case 'a':
			if (record[record.size() - 1] == 0)
			{
				record.pop_back();
				Mat temp = imread("women.jpg");
				pointList0.pop_back();
				pointList0.pop_back();
				for (int i = 0; i < pointList0.size(); i += 2)
					line(temp, pointList0[i], pointList0[i + 1], Scalar(0, 0, 255), 2);
				imshow("Image0", temp);
				temp0 = temp;
			}
			else //1
			{
				record.pop_back();
				Mat temp = imread("cheetah.jpg");
				pointList1.pop_back();
				pointList1.pop_back();
				for (int i = 0; i < pointList1.size(); i += 2)
					line(temp, pointList1[i], pointList1[i + 1], Scalar(0, 0, 255), 2);
				imshow("Image1", temp);
				temp1 = temp;
			}
			break;
		case 'q':
			for (int i = 0; i <= 10; i++)
			{
				float j = i / 10.0;
				Mat img00 = img0.clone();
				Mat img000 = img0.clone();
				Mat img11 = img1.clone();
				Mat img111 = img1.clone();
				newLines(&img0, 0, j);
				newLines(&img1, 1, j);
				WarpImage(&img00, &img000, 0);
				WarpImage(&img11, &img111, 1);
				addWeighted(img000, 1 - j, img111, j, 0.0, dst);
				cv::String filename = format("imgBlur%d.png", i);
				cv::imwrite(filename, dst);
				newpointList0.erase(newpointList0.begin(), newpointList0.end());
				newpointList1.erase(newpointList1.begin(), newpointList1.end());
			}

			//create video
			frame = cv::imread("imgBlur0.png");
			writer = cv::VideoWriter("out.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 3, frame.size());
			for (int i = 0; i <= 10; i++)
			{
				cv::String filename = format("imgBlur%d.png", i);
				frame = cv::imread(filename);
				writer << frame;
			}
			writer.release();
			break;
		default:
			break;
		}
		key = cvWaitKey(0);
	}
}