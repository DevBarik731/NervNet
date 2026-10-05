#pragma once

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
        if(A.data.size()==0 || A.data[0].size()==0){
            cout<<"Can't Initialize the Layer Invalid Input"<<endl;
            return;
        }
        int n=A.data[0].size();
        B.zero(1,m);
        W.zero(n,m);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                W.data[i][j]=dist(gen);
            }
        }
        Act=s;
    }
    matrix flow(matrix &X){
        Z=X*W+B;
        matrix A=Z;
        if(Act=="ReLU") A.ReLU();
        else if(Act=="sigmoid") A.sigmoid();
        else if(Act=="softmax") A.softmax();
        return A;
    }
};

class NeuralNetwork{
    public:
    matrix tpp;
    vector<matrix> A;
    vector<Layer> L;
    vector<matrix> d_W;
    vector<matrix> Loss={tpp};
    vector<matrix> d_B;
    vector<matrix> X,Y;
    vector<matrix> V_W,V_B,S_W,S_B;
    double alpha,beta_1=0.9,beta_2=0.999;
    double ep=1e-8;

    void fit(matrix &_X,matrix &_Y,double _alpha,int batch_size=1){
        if(batch_size<=0){
            cout<<"Can't be a non-positive batch size"<<endl;
        }
        alpha=_alpha;
        if(!_X.data.size() || !_X.data[0].size() || !_Y.data.size() || !_Y.data[0].size()){
            cout<<"Invalid Data for Neural Network"<<endl;
            return;
        }
        int n_x=_X.data[0].size();
        int m_x=_X.data.size();
        int n_y=_Y.data[0].size();
        int m_y=_Y.data.size();
        if(m_x!=m_y){
            cout<<"Invalid Data for Neural Network"<<endl;
            return;
        }
        // if(n_y!=1){
        //     cout<<"Invalid Data for Neural Network"<<endl;
        //     return;
        // }
        // X=_X;
        // Y=_Y;
        matrix temp_X;
        for(int i=0;i<_X.data.size();i++){
            temp_X.data.push_back(_X.data[i]);
            if((i+1)%batch_size==0){
                X.push_back(temp_X);
                temp_X={};
            }
        }
        if(temp_X.data.size()) X.push_back(temp_X);

        matrix temp_Y;
        for(int i=0;i<_Y.data.size();i++){
            temp_Y.data.push_back(_Y.data[i]);
            if((i+1)%batch_size==0){
                Y.push_back(temp_Y);
                temp_Y={};
            }
        }
        if(temp_Y.data.size()) Y.push_back(temp_Y);

        A.push_back(X[0]);
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
        
        d_W.push_back(tpp);
        V_W.push_back(tpp);
        S_W.push_back(tpp);

        
        d_B.push_back(tpp);
        V_B.push_back(tpp);
        S_B.push_back(tpp);

        Loss.push_back(tpp);
    }

    void ForwardProp(){
        for(int i=1;i<A.size();i++){
            A[i]=(L[i-1].flow(A[i-1]));
        }
    }
    void Adam(int it){
        
        V_W[it]=V_W[it]*beta_1;
       
        V_B[it]=V_B[it]*beta_1;
       
        S_W[it]=S_W[it]*beta_2;
        
        S_B[it]=S_B[it]*beta_2;

        matrix w_sq=matSquare(d_W[it]);
        matrix b_sq=matSquare(d_B[it]);


        d_W[it]=d_W[it]*(1.0-beta_1);
       
        d_B[it]=d_B[it]*(1.0-beta_1);
        
        w_sq=w_sq*(1.0-beta_2);
        
        b_sq=b_sq*(1.0-beta_2);

        V_W[it]=V_W[it]+d_W[it];
        V_B[it]=V_B[it]+d_B[it];
        S_W[it]=S_W[it]+w_sq;
        S_B[it]=S_B[it]+b_sq;
        
        for(int i=0;i<d_W[it].data.size();i++){
            for(int j=0;j<d_W[it].data[0].size();j++){
                d_W[it].data[i][j]=alpha*(V_W[it].data[i][j])/((sqrt(S_W[it].data[i][j]))+ep);
            }
        }
        for(int i=0;i<d_B[it].data.size();i++){
            for(int j=0;j<d_B[it].data[0].size();j++){
                d_B[it].data[i][j]=alpha*(V_B[it].data[i][j])/((sqrt(S_B[it].data[i][j]))+ep);
            }
        }
    }
    void UpdateWeights(){

        for(int i=1;i<A.size();i++){
            Adam(i-1);
            L[i-1].W=L[i-1].W-d_W[i-1];
            L[i-1].B=L[i-1].B-d_B[i-1];
        }
    }

    
    void BackProp(matrix &Y_mini){
        int m=Y_mini.data.size();
        int l=L.size();
        if(L[l-1].Act=="softmax"){
            Loss[l]=A[l]-Y_mini;
            Loss[l]=Loss[l]*(1.0/m);
        }
        else{
            Loss[l]=A[l]-Y_mini;
            Loss[l]=Loss[l]*(2.0/m);

            if(L[l-1].Act=="ReLU") Loss[l]=D_ReLU(Loss[l],L[l-1].Z);
            if(L[l-1].Act=="sigmoid") Loss[l]=D_sigmoid(Loss[l],L[l-1].Z);
        }

        for(int i=l-1;i>=0;i--){

            d_W[i]=matT(A[i])*Loss[i+1];

            if(V_W[i].data.empty() || V_W[i].data[0].empty()) V_W[i].zero(d_W[i].data.size(),d_W[i].data[0].size());
            if(S_W[i].data.empty() || S_W[i].data[0].empty()) S_W[i].zero(d_W[i].data.size(),d_W[i].data[0].size());



            if(i>0) Loss[i]=Loss[i+1]*matT(L[i].W);

            if(i>0 && L[i-1].Act=="ReLU") Loss[i]=D_ReLU(Loss[i],L[i-1].Z);

            d_B[i].zero(1,Loss[i+1].data[0].size());

            if(V_B[i].data.empty() || V_B[i].data[0].empty()) V_B[i].zero(1,Loss[i+1].data[0].size());
            if(S_B[i].data.empty() || S_B[i].data[0].empty()) S_B[i].zero(1,Loss[i+1].data[0].size());

            for(int j=0;j<Loss[i+1].data[0].size();j++){
                for(int k=0;k<Loss[i+1].data.size();k++){
                    d_B[i].data[0][j]+=Loss[i+1].data[k][j];
                }
            }
        }
    }
    void train(int epoch){
        for(int it=0;it<epoch;it++){
            for(int k=0;k<X.size();k++){
                A[0]=X[k];
                ForwardProp();
                BackProp(Y[k]);
                UpdateWeights();
            }
            
        }
    }
    matrix predict(matrix &X_test){
        if(!X_test.data.size() || !X_test.data[0].size()){
            cout<<"Invalid data set : put a valid dataset to predict result"<<endl;
            return {};
        }
        int X_test_m=X_test.data[0].size();

        if(!L.size() || X_test_m!=L[0].W.data.size()){
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