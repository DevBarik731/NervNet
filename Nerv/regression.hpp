#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "matrix.hpp"
using namespace std;



// LinearRegression Class
class LinearRegression{
    public:
    matrix W,d_w,B,X,Y;
    int n,m;
    void fit(matrix &x,matrix &y){
        int n_x=x.data[0].size();
        int m_x=x.data.size();
        int n_y=y.data[0].size();
        int m_y=y.data.size();
        if(m_x!=m_y){
            cout<<"Invalid Dataset"<<endl;
            return;
        }
        if(n_y!=1){
            cout<<"Invalid Dataset"<<endl;
            return;
        }
        n=n_x;
        m=m_x;
        X=x;
        Y=y;
        B.zero(1,1);
        W.zero(n,1);
        d_w.zero(n,1);
    }
    void train(int steps,double alpha){
        matrix X_T=matT(X);
        for(int i=0;i<steps;i++){
            matrix Y_pred=X*W+B;
            matrix Y_delta=Y_pred-Y;
            d_w=X_T*Y_delta;
            d_w=(d_w*((1.0*alpha)/m));
            W=W-d_w;
            double sum=0;
            for(int i=0;i<m;i++){
                sum+=Y_delta.data[i][0];
            }
            sum=(sum*alpha)/(1.0*m);
            B.data[0][0]-=(sum);
        }
    }
    matrix predict(matrix &x){
        int n_x=x.data[0].size();
        int m_x=x.data.size();
        if(n_x!=n){
            cout<<"Invalid dataset"<<endl;
            return {};
        }
        matrix Y_pred=x*W+B;
        return Y_pred;
    }
    
};

// Logistic Regression implementation
class LogisticRegression{
public:
    matrix W,d_w,B,X,Y;
    int n,m;
    void fit(matrix &x,matrix &y){
        int n_x=x.data[0].size();
        int m_x=x.data.size();
        int n_y=y.data[0].size();
        int m_y=y.data.size();
        if(m_x!=m_y){
            cout<<"Invalid Dataset"<<endl;
            return;
        }
        if(n_y!=1){
            cout<<"Invalid Dataset"<<endl;
            return;
        }
        for(auto &a:y.data){
            for(auto &b:a){
                if(b!=0 && b!=1){
                    cout<<"Only 0/1 y is valid in Logistic Regression"<<endl;
                    return;
                }
            }
        }
        n=n_x;
        m=m_x;
        X=x;
        Y=y;
        B.zero(1,1);
        W.zero(n,1);
        d_w.zero(n,1);
    }
    void train(int steps,double alpha){
        matrix X_T=matT(X);
        for(int i=0;i<steps;i++){
            matrix Y_pred=X*W+B;
            Y_pred.sigmoid();
            matrix Y_delta=Y_pred-Y;
            d_w=X_T*Y_delta;
            d_w=d_w*((1.0*alpha)/m);
            W=W-d_w;
            double sum=0;
            for(int i=0;i<m;i++){
                sum+=Y_delta.data[i][0];
            }
            sum=(sum*alpha)/(1.0*m);
            B.data[0][0]-=(sum);
        }
    }
    matrix predict(matrix &x){
        int n_x=x.data[0].size();
        int m_x=x.data.size();
        if(n_x!=n){
            cout<<"Invalid dataset"<<endl;
            return {};
        }
        matrix Y_pred=x*W+B;
        Y_pred.sigmoid();
        Y_pred.sigToBin();
        return Y_pred;
    }
};

