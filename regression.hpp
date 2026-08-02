#ifndef REGRESSION_HPP
#define REGRESSION_HPP

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
        int n_x=x[0].size();
        int m_x=x.size();
        int n_y=y[0].size();
        int m_y=y.size();
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
        zero(B,1,1);
        zero(W,n,1);
        zero(d_w,n,1);
    }
    void train(int steps,double alpha){
        matrix X_T=matT(X);
        for(int i=0;i<steps;i++){
            matrix Y_pred=matadd(matmul(X,W),B);
            matrix Y_delta=matsub(Y_pred,Y);
            d_w=matmul(X_T,Y_delta);
            matScalar(d_w,(1.0*alpha)/m);
            W=matsub(W,d_w);
            double sum=0;
            for(int i=0;i<m;i++){
                sum+=Y_delta[i][0];
            }
            sum=(sum*alpha)/(1.0*m);
            B[0][0]-=(sum);
        }
    }
    matrix predict(matrix &x){
        int n_x=x[0].size();
        int m_x=x.size();
        if(n_x!=n){
            cout<<"Invalid dataset"<<endl;
            return {};
        }
        matrix Y_pred=matadd(matmul(x,W),B);
        return Y_pred;
    }
    
};

// Logistic Regression implementation
class LogisticRegression{
public:
    matrix W,d_w,B,X,Y;
    int n,m;
    void fit(matrix &x,matrix &y){
        int n_x=x[0].size();
        int m_x=x.size();
        int n_y=y[0].size();
        int m_y=y.size();
        if(m_x!=m_y){
            cout<<"Invalid Dataset"<<endl;
            return;
        }
        if(n_y!=1){
            cout<<"Invalid Dataset"<<endl;
            return;
        }
        for(auto &a:y){
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
        zero(B,1,1);
        zero(W,n,1);
        zero(d_w,n,1);
    }
    void train(int steps,double alpha){
        matrix X_T=matT(X);
        for(int i=0;i<steps;i++){
            matrix Y_pred=matadd(matmul(X,W),B);
            sigmoid(Y_pred);
            matrix Y_delta=matsub(Y_pred,Y);
            d_w=matmul(X_T,Y_delta);
            matScalar(d_w,(1.0*alpha)/m);
            W=matsub(W,d_w);
            double sum=0;
            for(int i=0;i<m;i++){
                sum+=Y_delta[i][0];
            }
            sum=(sum*alpha)/(1.0*m);
            B[0][0]-=(sum);
        }
    }
    matrix predict(matrix &x){
        int n_x=x[0].size();
        int m_x=x.size();
        if(n_x!=n){
            cout<<"Invalid dataset"<<endl;
            return {};
        }
        matrix Y_pred=matadd(matmul(x,W),B);
        sigmoid(Y_pred);
        for(auto &a:Y_pred){
            for(auto &b:a){
                if(b>=0.5) b=1;
                else b=0;
            }
        }
        return Y_pred;
    }
};


#endif