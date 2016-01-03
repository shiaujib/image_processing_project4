#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include<iostream>
#include<fstream>
#include<math.h>
#define pi 3.14159
using namespace std;
using namespace cv;
Mat matSrc,matDst,matNormalize;

void init(){
	matSrc=Mat(Size(512,512),CV_64FC1);
	matDst=Mat(Size(512,512),CV_64FC1);
	matNormalize=Mat(Size(512,512),CV_8UC1);
	


}

void readFile(){
	ifstream fin("cof.txt");
	std::string line;
	char value[100];
	double dvalue;
	int num=0,rows=0,cols=0;
	int ln=0;
	/*while(fin.getline(value,sizeof(value))){
		dvalue=atof(value);
		cout<<value<<"  "<<num<<endl;
	}*/
	while(fin>>value){
		num++;
		dvalue=atof(value);
		matSrc.at<double>(rows,cols++)=dvalue;
		if(cols==512){
			rows++;
			cols=0;
		}
	}
	cout<<"total "<<num<<endl<<"rows "<<rows<<endl;

}
void IHWT(double *array,int len){
	double tmp[len];
	int index;
	int half=len/2;
	for(int i=0;i<half;i++){
		index=(i<<1);
		tmp[index]=(double)(array[i]+array[i+half]);
		tmp[index+1]=(double)(array[i]-array[i+half]);
	}
	for(int i=0;i<len;i++)
		array[i]=tmp[i];
}



void DIHWT(Mat matin,Mat matout,int iter){
	int div;
	int rows=matin.rows;
	int cols=matin.cols;
	int trows,tcols;
	for(int i=0;i<rows;i++)           //initialize
		for(int j=0;j<cols;j++){
			matout.at<double>(i,j)=matin.at<double>(i,j);
	        }


	for(int k=iter-1;k>=0;k--){
		div=1<<k;
		trows=rows/div;
		tcols=cols/div;
		double colArray[tcols];
		double rowArray[trows];
		for(int i=0;i<tcols;i++){
			for(int j=0;j<tcols;j++)
				colArray[j]=matout.at<double>(j,i);
			IHWT(colArray,tcols);


			for(int j=0;j<trows;j++){
				matout.at<double>(j,i)=colArray[j];
			}
		}

		for(int i=0;i<trows;i++){
			for(int j=0;j<trows;j++)
				rowArray[j]=matout.at<double>(i,j);
			IHWT(rowArray,trows);


			for(int j=0;j<trows;j++){
				matout.at<double>(i,j)=rowArray[j];
			}
		}
	}
}

void normalize(Mat matin,Mat matout){
	int cols=matin.cols;
	int rows=matin.rows;
	double max=0,min=1000,value,scale;
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			value=matin.at<double>(j,i);
			//cout<<value<<endl;
			if(value>max)
				max=value;
			if(value<min)
				min=value;
			}
	}
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			scale=(matin.at<double>(j,i)-min)*255.0/(max-min);
			matout.at<uchar>(j,i)=scale;
		}
	}
	cout<<"max="<<max<<endl;
	cout<<"min="<<min<<endl;
}
	
		


int main(){
	init();
	readFile();
	imshow("img",matSrc);
	waitKey(0);
	DIHWT(matSrc,matDst,3);
	normalize(matDst,matNormalize);
	imshow("result",matNormalize);
	waitKey(0);
	

}

	
