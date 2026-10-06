#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
using namespace std;
// using 2D vector as matrix
class matrix{
    public:
    vector<vector<double>> data;
    void display(){
        int n=0,m=0;
        n=data.size();
        if(n) m=data[0].size();
        if(!n || !m){
            cout<<"Empty Matrix"<<endl;
            return;
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++) cout<<data[i][j]<<" ";
            cout<<"\n";
        }
    }

    void zero(int a,int b){
        data=vector<vector<double>>(a,vector<double>(b,0));
    }

    void sigmoid(){
        for(auto &x:data){
            for(auto &y:x){
            if(y>=0) y=1.0/(1.0+exp(-y));
            else{
                double t=exp(y);
                y=t/(1+t);
            }
            }
        }
    }

    void sigToBin(){
        for(auto &x:data){
            for(auto &y:x){
            if(y>=0.5) y=1;
            else y=0; 
            }
        }
    }

    void ReLU(){
        for(auto &x:data){
            for(auto &y:x){
                y=max(y,0.0);
            }
        }
    }
    void softmax(){
        int n=0,m=0;
        n=data.size();
        if(n) m=data[0].size();
        if(!n || !m){
            cout<<"Empty Matrix"<<endl;
            return;
        }
        for(int i=0;i<n;i++){
            double sum=0;
            double mx=data[i][0];
            for(int j=0;j<m;j++) mx=max(mx,data[i][j]);
            for(int j=0;j<m;j++){
                data[i][j]=exp(data[i][j]-mx);
                sum+=data[i][j];
            }
            for(int j=0;j<m;j++) data[i][j]/=sum;
        }
    }
    
    matrix sofToClass(){
        matrix z;
        int n=0,m=0;
        n=data.size();
        if(n) m=data[0].size();
        if(!n || !m){
            cout<<"Empty Matrix"<<endl;
            return z;
        }
        z.zero(n,1);
        for(int i=0;i<n;i++){
            z.data[i][0]=0;
            for(int j=0;j<m;j++){
                if(data[i][j]>data[i][z.data[i][0]]) z.data[i][0]=j;
            }
        }
        return z;
    }
};
matrix nullmat;
// subtraction of matrices
matrix operator-(const matrix& a, const matrix& b){
    int a_row=a.data.size();
    int a_col=a.data[0].size();
    int b_row=b.data.size();
    int b_col=b.data[0].size();
    matrix result;
    result.zero(a_row,a_col);
    if((a_col!=b_col && b_col!=1) || (a_row!=b_row && b_row!=1)){
        cout<<"Invalid"<<endl;
        return nullmat;
    }
    for(int i=0;i<a_row;i++){
        for(int j=0;j<a_col;j++){
            result.data[i][j]=a.data[i][j]-b.data[i%b_row][j%b_col];
        }
    }
    return result;

}

// addition of matrices
matrix operator+(const matrix& a, const matrix& b){
    int a_row=a.data.size();
    int a_col=a.data[0].size();
    int b_row=b.data.size();
    int b_col=b.data[0].size();
    matrix result;
    result.zero(a_row,a_col);
    if((a_col!=b_col && b_col!=1) || (a_row!=b_row && b_row!=1)){
        cout<<"Invalid"<<endl;
        return nullmat;
    }
    for(int i=0;i<a_row;i++){
        for(int j=0;j<a_col;j++){
            result.data[i][j]=a.data[i][j]+b.data[i%b_row][j%b_col];
        }
    }
    return result;
}


//multiplication of Matrices
matrix operator*(const matrix& a,const matrix &b){
    matrix result;
    int a_row=a.data.size();
    int a_col=a.data[0].size();
    int b_row=b.data.size();
    int b_col=b.data[0].size();
    if(a_col!=b_row){
        cout<<"invalid matrix multiplication"<<endl;
        return result;
    }
    result.zero(a_row,b_col);
    for(int i=0;i<a_row;i++){
        for(int j=0;j<b_col;j++){
            double sum=0;
            for(int it=0;it<a_col;it++) sum+=(a.data[i][it]*b.data[it][j]);
            result.data[i][j]=sum;
        }
    }
    return result;
}

//multiplying scalar with matrix
matrix operator*(const matrix &a,double k){
    matrix result=a;
    for(auto &x:result.data){
        for(auto &y:x){
            y*=k;
        }
    }
    return result;
}

matrix operator*(double k,const matrix &a){
    matrix result=a;
    for(auto &x:result.data){
        for(auto &y:x){
            y*=k;
        }
    }
    return result;
}

matrix matSquare(const matrix &a){
    matrix z=a;
    for(auto &x:z.data){
        for(auto &y:x){
            y*=y;
        }
    }
    return z;
}

//taking Transpose
matrix matT(const matrix& _a){
    vector<vector<double>> a=_a.data;
    matrix result;
    result.zero(a[0].size(),a.size());
    for(int i=0;i<(a.size());i++){
        for(int j=0;j<a[0].size();j++){
            result.data[j][i]=a[i][j];
        }
    }
    return result;
}


// reading Numeric CSV as a matrix;
matrix readCSV(const string &filename) {
    ifstream file(filename);
    matrix dataMat;
    string line;
    getline(file,line);
    
    while (getline(file, line)){
        stringstream ss(line);
        string cell;
        vector<double> row;

        while(getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        dataMat.data.push_back(row);
    }
    return dataMat;
}

// splitting the dataset into features and result
void split(const matrix& A,matrix &X,matrix &Y,int k){
    int n=A.data.size();
    int m=A.data[0].size();
    if(k<=0 || m-k<=0){
        cout<<"Can't Be Splited"<<endl;
        return;
    }
    X.zero(n,m-k);
    Y.zero(n,k);
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(j>=m-k) Y.data[i][j+k-m]=A.data[i][j];
            else X.data[i][j]=A.data[i][j];
        }
    }
}

// Rectified Linear Unit

matrix D_ReLU(matrix &A,matrix &a){
    matrix z=a;
    for(int i=0;i<a.data.size();i++){
        for(int j=0;j<a.data[0].size();j++){
            if(a.data[i][j]>0) z.data[i][j]=1;
            else z.data[i][j]=0;
            z.data[i][j]*=A.data[i][j];
        }
    }
    return z;
    
}

matrix D_sigmoid(matrix &A,matrix &a){
    matrix z=a;
    for(int i=0;i<a.data.size();i++){
        for(int j=0;j<a.data[0].size();j++){
            double t;
            if(a.data[i][j]>=0) t=exp(-a.data[i][j]);
            else t=exp(a.data[i][j]);
            z.data[i][j]=(t/((t+1)*(t+1)));
            z.data[i][j]*=A.data[i][j];
        }
    }
    return z;
}

double RMSE(matrix &Y,matrix &Y_pred){
    double error=0.0;
    double m=Y.data.size();
    for(int i=0;i<Y.data.size();i++){
        for(int j=0;j<Y.data[0].size();j++){
            error+=((Y.data[i][j]-Y_pred.data[i][j])*(Y.data[i][j]-Y_pred.data[i][j]));
        }
    }
    error=(error*(1.0)/(m+0.00001));
    return sqrt(error);
}

// calculating accuracy
double accuracy(matrix &Y,matrix &Y_pred){
    double accuracy=0.0;
    double m=Y.data.size();
    for(int i=0;i<Y.data.size();i++){
        for(int j=0;j<Y.data[0].size();j++){
            if(Y.data[i][j]==Y_pred.data[i][j]) accuracy++;
        }
    }
    return (accuracy)*1.0/m;
}


// calculating Z-score for dataset
class Zscale{
    public:
    vector<double> mean,var,stdv;

    void fit(matrix &a){
    if (a.data.empty() || a.data[0].empty()) {
        cout<<"Invalid Data set cannot be fitted into scale"<<endl;
        return;
    }
    int n=a.data[0].size();
    int m=a.data.size();
    mean=vector<double>(n,0);
    var=vector<double>(n,0);
    stdv=vector<double>(n,0);
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            mean[j]+=a.data[i][j];
        }
        mean[j]=mean[j]/(1.0*m);
    }
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            var[j]+=((a.data[i][j]-mean[j])*(a.data[i][j]-mean[j]));
        }
        var[j]=(var[j]/(1.0*m));
    }
    for(int j=0;j<n;j++) stdv[j]=sqrt(var[j]);
}


void scale(matrix &a){
    if(a.data.size()==0 || a.data[0].size()==0){
        cout<<"Invalid DataSet Can't be scaled"<<endl;
        return;
    }
    if(mean.size()==0){
        cout<<"No fitted data on scale"<<endl;
        return;
    }
    int n=a.data[0].size();
    int m=a.data.size();
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            if(stdv[j]==0) a.data[i][j]=0;
            else a.data[i][j]=(a.data[i][j]-mean[j])/stdv[j];
        }
    }
}

void descale(matrix &a){
    if(a.data.size()==0 || a.data[0].size()==0){
        cout<<"Invalid DataSet Can't be scaled"<<endl;
        return;
    }
    if(mean.size()==0){
        cout<<"No fitted data on scale"<<endl;
        return;
    }
    int n=a.data[0].size();
    int m=a.data.size();
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            a.data[i][j]=a.data[i][j]*stdv[j]+mean[j];
        }
    }
}
};
