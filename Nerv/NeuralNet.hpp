#ifndef NN_HPP
#define NN_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include "matrix.hpp"
using namespace std;

class Layer{
    public:
    matrix W,B,Z;
    string Act;
    void fit(int m,matrix &A,string s="Linear"){
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-0.5, 0.5);
        if(A.size()==0 || A[0].size()==0){
            cout<<"Can't Initialize the Layer Invalid Input"<<endl;
            return;
        }
        int n=A[0].size();
        zero(B,1,m);
        zero(W,n,m);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                W[i][j]=dist(gen);
            }
        }
        Act=s;
    }
    matrix flow(matrix &X){
        Z=matadd(matmul(X,W),B);
        matrix A=Z;
        if(Act=="ReLU") ReLU(A);
        else if(Act=="sigmoid") sigmoid(A);
        return A;
    }
};

class NeuralNetwork{
    public:
    vector<matrix> A;
    vector<Layer> L;
    vector<matrix> d_W;
    vector<matrix> Loss={{}};
    vector<matrix> d_B;
    matrix X,Y;
    double alpha;
    void fit(matrix &_X,matrix &_Y,double _alpha){
        alpha=_alpha;
        if(!_X.size() || !_X[0].size() || !_Y.size() || !_Y[0].size()){
            cout<<"Invalid Data for Neural Network"<<endl;
            return;
        }
        int n_x=_X[0].size();
        int m_x=_X.size();
        int n_y=_Y[0].size();
        int m_y=_Y.size();
        if(m_x!=m_y){
            cout<<"Invalid Data for Neural Network"<<endl;
            return;
        }
        if(n_y!=1){
            cout<<"Invalid Data for Neural Network"<<endl;
            return;
        }
        X=_X;
        Y=_Y;
        A.push_back(X);
    }

    void addLayer(int m,string s="Linear"){
        if(X.size()==0){
            cout<<"Before Adding Layers Add Data"<<endl;
            return;
        }
        Layer lyr;
        lyr.fit(m,A.back(),s);
        matrix a=lyr.flow(A.back());
        L.push_back(lyr);
        A.push_back(a);
        d_W.push_back({{}});
        d_B.push_back({{}});
        Loss.push_back({{}});
    }

    void ForwardProp(){
        for(int i=1;i<A.size();i++){
            matScalar(d_W[i-1],alpha);
            matScalar(d_B[i-1],alpha);
            L[i-1].W=matsub(L[i-1].W,d_W[i-1]);
            L[i-1].B=matsub(L[i-1].B,d_B[i-1]);
            A[i]=(L[i-1].flow(A[i-1]));
        }
    }
    void BackProp(){
        int m=Y.size();
        int l=L.size();
        Loss[l]=matsub(A[l],Y);
        matScalar(Loss[l],(2.0/m));

        if(L[l-1].Act=="ReLU") Loss[l]=D_ReLU(Loss[l],L[l-1].Z);
        if(L[l-1].Act=="sigmoid") Loss[l]=D_sigmoid(Loss[l],L[l-1].Z);
        for(int i=l-1;i>=0;i--){

            d_W[i]=matmul(matT(A[i]),Loss[i+1]);

            if(i>0) Loss[i]=matmul(Loss[i+1],matT(L[i].W));

            if(i>0 && L[i-1].Act=="ReLU") Loss[i]=D_ReLU(Loss[i],L[i-1].Z);

            zero(d_B[i],1,Loss[i+1][0].size());

            for(int j=0;j<Loss[i+1][0].size();j++){
                for(int k=0;k<Loss[i+1].size();k++){
                    d_B[i][0][j]+=Loss[i+1][k][j];
                }
            }
        }
    }
    void train(int steps){
        for(int it=0;it<steps;it++){
            BackProp();
            ForwardProp();
        }
    }
    matrix predict(matrix &X_test){
        if(!X_test.size() || !X_test[0].size()){
            cout<<"Invalid data set : put a valid dataset to predict result"<<endl;
            return {};
        }
        int X_test_m=X_test[0].size();

        if(!L.size() || X_test_m!=L[0].W.size()){
            cout<<"Invalid data set : put a valid dataset to predict result"<<endl;
            return {};
        }
        matrix Y_out=X_test;
        for(int i=1;i<A.size();i++){
            Y_out=L[i-1].flow(Y_out);
        }
        return Y_out;
    }

    
};
#endif