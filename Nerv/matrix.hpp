#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
using namespace std;
// using 2D vector as matrix
typedef vector<vector<double>> matrix;

// assigning zeros to matrix
void zero(matrix &A,int a,int b){
    A=matrix(a,vector<double>(b,0));
}

// subtraction of matrices
matrix matsub(const matrix&a,const matrix&b){
    int a_row=a.size();
    int a_col=a[0].size();
    int b_row=b.size();
    int b_col=b[0].size();
    matrix result;
    result=matrix(a_row,vector<double>(a_col,0));
    if((a_col!=b_col && b_col!=1) || (a_row!=b_row && b_row!=1)){
        cout<<"Invalid"<<endl;
        return {};
    }
    for(int i=0;i<a_row;i++){
        for(int j=0;j<a_col;j++){
            result[i][j]=a[i][j]-b[i%b_row][j%b_col];
        }
    }
    return result;

}

// addition of matrices
matrix matadd(const matrix&a,const matrix&b){
    int a_row=a.size();
    int a_col=a[0].size();
    int b_row=b.size();
    int b_col=b[0].size();
    matrix result;
    result=matrix(a_row,vector<double>(a_col,0));
    if((a_col!=b_col && b_col!=1) || (a_row!=b_row && b_row!=1)){
        cout<<"Invalid"<<endl;
        return {};
    }
    for(int i=0;i<a_row;i++){
        for(int j=0;j<a_col;j++){
            result[i][j]=a[i][j]+b[i%b_row][j%b_col];
        }
    }
    return result;
}


//multiplication of Matrices
matrix matmul(const matrix& a,const matrix &b){
    matrix result;
    int a_row=a.size();
    int a_col=a[0].size();
    int b_row=b.size();
    int b_col=b[0].size();
    if(a_col!=b_row){
        cout<<"invalid matrix multiplication"<<endl;
        return result;
    }
    result=matrix(a_row,vector<double>(b_col,0));
    for(int i=0;i<a_row;i++){
        for(int j=0;j<b_col;j++){
            double sum=0;
            for(int it=0;it<a_col;it++) sum+=(a[i][it]*b[it][j]);
            result[i][j]=sum;
        }
    }
    return result;
}

//multiplying scalar with matrix
void matScalar(matrix &a,double k){
    for(auto &x:a){
        for(auto &y:x){
            y*=k;
        }
    }
}


//taking Transpose
matrix matT(const matrix& a){
    matrix result(a[0].size(),vector<double>(a.size(),0));
    for(int i=0;i<(a.size());i++){
        for(int j=0;j<a[0].size();j++){
            result[j][i]=a[i][j];
        }
    }
    return result;
}

//Displaying matrix
void matDisplay(matrix &a){
    int n=a.size();
    int m=a[0].size();
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++) cout<<a[i][j]<<" ";
        cout<<"\n";
    }
}

// reading Numeric CSV as a matrix;
matrix readCSV(const string &filename) {
    ifstream file(filename);
    matrix data;
    string line;
    getline(file,line);
    
    while (getline(file, line)){
        stringstream ss(line);
        string cell;
        vector<double> row;

        while(getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    return data;
}

// splitting the dataset into features and result
void split(const matrix& A,matrix &X,matrix &Y){
    int n=A.size();
    int m=A[0].size();
    if(m<=1){
        cout<<"Can't Be Splited"<<endl;
        return;
    }
    zero(X,n,m-1);
    zero(Y,n,1);
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(j==m-1) Y[i][0]=A[i][j];
            else X[i][j]=A[i][j];
        }
    }
}
// sigmoids function
void sigmoid(matrix &a){
    for(auto &x:a){
        for(auto &y:x){
           if(y>=0) y=1.0/(1.0+exp(-y));
           else{
            double t=exp(y);
            y=t/(1+t);
           }
        }
    }
}

void sigToBin(matrix &a){
    for(auto &x:a){
        for(auto &y:x){
           if(y>=0.5) y=1;
           else y=0; 
        }
    }
}

// Rectified Linear Unit
void ReLU(matrix &a){
    for(auto &x:a){
        for(auto &y:x){
            y=max(y,0.0);
        }
    }
}

matrix D_ReLU(matrix &A,matrix &a){
    matrix z=a;
    for(int i=0;i<a.size();i++){
        for(int j=0;j<a[0].size();j++){
            if(a[i][j]>0) z[i][j]=1;
            else z[i][j]=0;
            z[i][j]*=A[i][j];
        }
    }
    return z;
    
}

matrix D_sigmoid(matrix &A,matrix &a){
    matrix z=a;
    for(int i=0;i<a.size();i++){
        for(int j=0;j<a[0].size();j++){
            double t;
            if(a[i][j]>=0) t=exp(-a[i][j]);
            else t=exp(a[i][j]);
            z[i][j]=(t/((t+1)*(t+1)));
            z[i][j]*=A[i][j];
        }
    }
    return z;
}

double RMSE(matrix &Y,matrix &Y_pred){
    double error=0.0;
    double m=Y.size();
    for(int i=0;i<Y.size();i++){
        for(int j=0;j<Y[0].size();j++){
            error+=((Y[i][j]-Y_pred[i][j])*(Y[i][j]-Y_pred[i][j]));
        }
    }
    error=(error*(1.0)/(m+0.00001));
    return sqrt(error);
}

// calculating accuracy
double accuracy(matrix &Y,matrix &Y_pred){
    double accuracy=0.0;
    double m=Y.size();
    for(int i=0;i<Y.size();i++){
        for(int j=0;j<Y[0].size();j++){
            if(Y[i][j]==Y_pred[i][j]) accuracy++;
        }
    }
    return (accuracy)*1.0/m;
}


// calculating Z-score for dataset
class Zscale{
    public:
    vector<double> mean,var,stdv;

    void fit(matrix &a){
    if (a.empty() || a[0].empty()) {
        cout<<"Invalid Data set cannot be fitted into scale"<<endl;
        return;
    }
    int n=a[0].size();
    int m=a.size();
    mean=vector<double>(n,0);
    var=vector<double>(n,0);
    stdv=vector<double>(n,0);
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            mean[j]+=a[i][j];
        }
        mean[j]=mean[j]/(1.0*m);
    }
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            var[j]+=((a[i][j]-mean[j])*(a[i][j]-mean[j]));
        }
        var[j]=(var[j]/(1.0*m));
    }
    for(int j=0;j<n;j++) stdv[j]=sqrt(var[j]);
}


void scale(matrix &a){
    if(a.size()==0 || a[0].size()==0){
        cout<<"Invalid DataSet Can't be scaled"<<endl;
        return;
    }
    if(mean.size()==0){
        cout<<"No fitted data on scale"<<endl;
        return;
    }
    int n=a[0].size();
    int m=a.size();
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            if(stdv[j]==0) a[i][j]=0;
            else a[i][j]=(a[i][j]-mean[j])/stdv[j];
        }
    }
}

void descale(matrix &a){
    if(a.size()==0 || a[0].size()==0){
        cout<<"Invalid DataSet Can't be scaled"<<endl;
        return;
    }
    if(mean.size()==0){
        cout<<"No fitted data on scale"<<endl;
        return;
    }
    int n=a[0].size();
    int m=a.size();
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            a[i][j]=a[i][j]*stdv[j]+mean[j];
        }
    }
}
};
