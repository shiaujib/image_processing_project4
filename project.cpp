#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include<iostream>
#include<math.h>
#include<math.h>
#define pi 3.14159
using namespace std;
using namespace cv;

Mat matSrc,matDst,matDst2,matH,matHT;

void init(String str){
        matSrc=imread(str,0);
        Mat matSrc=imread(str);
	matDst=Mat(matSrc.size(),CV_64FC1);
	matDst2=Mat(matSrc.size(),CV_8UC1);
	matH=Mat(matSrc.size(),CV_64FC1);
	matHT=Mat(matSrc.size(),CV_64FC1);


}

void productMatrix(Mat mata, Mat matb,Mat matout,int flag){
	int cols=mata.cols;
	int rows=mata.rows;
	double sum;
	if(flag==0){
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			for(int k=0;k<cols;k++){
				sum+=mata.at<double>(i,k)*matb.at<uchar>(k,j);
			}
			matout.at<double>(i,j)=sum;
			sum=0;
		}
	}
	}
	if(flag==1){
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			for(int k=0;k<cols;k++){
				sum+=mata.at<double>(i,k)*matb.at<double>(k,j);
			}
			matout.at<double>(i,j)=sum;
			sum=0;
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

void DHWT(double array[],int len){
	int j;
	double tmp[len];
	int half=len/2;
	for(int i=0;i<half;i++){
		j=i*2;
		tmp[i]=(double)(array[j]+array[j+1])/2;
		tmp[i+half]=(double)(array[j]-array[j+1])/2;
	}
	for(int i=0;i<len;i++)
		array[i]=tmp[i];
}
		

void DWT(Mat matin,Mat matout,int iter){
	int cols=matin.cols;
	int rows=matin.rows;
	int tcols,trows;
	double colA[cols];
	int n;
	for(int i=0;i<rows;i++)           //initialize
		for(int j=0;j<cols;j++){
			matout.at<double>(i,j)=matin.at<uchar>(i,j);
		}
			
	for(int k=0;k<iter;k++){
		n=1<<k;
		tcols=cols/n;   //each iteration column will divide by 2
		trows=rows/n;   
		double rowA[trows];
		double colA[tcols];
	for(int i=0;i<trows;i++){

		for(int j=0;j<trows;j++){
				rowA[j]=matout.at<double>(i,j);
			//cout<<"aa  "<<rowA[j]<<"      ";
		}
			DHWT(rowA,trows);

		for(int j=0;j<trows;j++)   //save the row value after trans into matrix
			matout.at<double>(i,j)=rowA[j];
	}


	//          cols part of haar trans

	for(int j=0;j<tcols;j++){

		for(int i=0;i<tcols;i++){
			colA[i]=matout.at<double>(i,j);
		}
			DHWT(colA,tcols);

		for(int i=0;i<tcols;i++)
			matout.at<double>(i,j)=colA[i];
	}
	}
}
			

			




void getHaarMatrix(Mat matin){
	int cols,rows;
	cols=matin.cols;
	rows=matin.rows;
	int p,q,k,N=cols;
	double z;
	double h[N][N];
	double ht[N][N];
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			p=0;
			q=i-pow(2,p)+1;
			if(p==0&&q!=0&&q!=1){
				p++;
				q=i-pow(2,p)+1;
				
			}
			while(p>0&&(q<1||q>pow(2,p))){
				p++;
				q=i-pow(2,p)+1;
			}
			z=(double)j/N;
		//	cout<<"z="<<z<<endl<<"p="<<p<<endl<<"q="<<q<<endl;
			if(z>=(q-1)/pow(2,p)&&z<(q-0.5)/pow(2,p))
				h[i][j]=pow(2,0.5*p);
			else if(z>=(q-0.5)/pow(2,p)&&z<q/pow(2,p))
				h[i][j]=-pow(2,0.5*p);
			
			else
				h[i][j]=0;
			
			if(i==0)
				h[i][j]=1;
		}
	}
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			h[i][j]/=pow(N,0.5);
			matH.at<double>(i,j)=h[i][j];		
		//	cout<<h[i][j]<<"\t";
		}
	//	cout<<endl;
	}
	cout<<endl<<endl;	
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			ht[j][i]=h[i][j];
			//matHT.at<double>(i,j)=ht[i][j];		
		}
	}
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			matHT.at<double>(i,j)=ht[i][j];		
		//	cout<<matHT.at<double>(i,j)<<"\t";
		}
	//cout<<endl;
	}
	imshow("H",matH);
	waitKey(0);
	imshow("TH",matHT);
	waitKey(0);
			
}

void writeFile(){
	FILE *fp;
	fp=fopen("cof.txt","w");
	for(int i=0;i<matDst.rows;i++){
		for(int j=0;j<matDst.cols;j++){
			fprintf(fp,"%lf ",matDst.at<double>(i,j));
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}


			
			
	


int main(){
	init("Fig0809(a).tif");
	DWT(matSrc,matDst,3);
	normalize(matDst,matDst2);
//	normalize(matSrc,matDst2,0,255,NORM_MINMAX,CV_8UC1);
	writeFile();
	imshow("state 1",matDst2);
	waitKey(0);
/*	getHaarMatrix(matSrc);
	productMatrix(matH,matSrc,matDst,1);
	normalize(matDst,matDst2);
	imshow("state 1",matDst);
	waitKey(0);
	productMatrix(matDst,matHT,matDst,1);
	normalize(matDst,matDst2,0,255,NORM_MINMAX,CV_8UC1);
//	normalize(matDst,matDst2);
	imshow("result",matDst2);
	waitKey(0);*/
	
	/*for(int i=100;i<150;i++)
		for(int j=0;j<25;j++)
			matDst.at<double>(j,i)=255;
	imshow("aaa",matDst);
	waitKey(0);
*/
	


}
