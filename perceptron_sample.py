# -*- coding: utf-8 -*-
import numpy as np
import random

class perceptron:
    

    
    def train(self):
        
        self.max_itera=100
        self.learning_rate=1
        self.num_instances=100
        self.theta=0
        self.weights=np.zeros(4)
        self.local_err=0
        self.global_err=0
        self.output=0
        self.outputs=np.zeros(self.num_instances)
        iteration=0
        #create train points
        
        x=np.zeros(self.num_instances)
        y=np.zeros(self.num_instances)
        z=np.zeros(self.num_instances)
        #class1
        for i in range(self.num_instances/2):
            x[i]=random.randrange(4,11)
            y[i]=random.randrange(5,8)
            z[i]=random.randrange(2,6)
        #class2   
        for i in range(self.num_instances/2,self.num_instances):
            x[i]=random.randrange(-2,3)
            y[i]=random.randrange(-4,5)
            z[i]=random.randrange(-6,2)
         
        for i in range(4):
            self.weights[i]=random.randrange(0,1)
            
        while(self.global_err!=0 and iteration<=self.max_itera):
            
            iteration+=1
            
            for i in range(self.num_instances):
                
                self.output=self.calculate_output(self.theta,self.weights,x[i],y[i],z[i])
                self.local_err=self.outputs[i]-self.output
                
                self.weights[0]+=self.learning_rate*self.local_err*x[i]
                self.weights[1]+=self.learning_rate*self.local_err*y[i]
                self.weights[2]+=self.learning_rate*self.local_err*z[i]
                self.weights[3]+=self.learning_rate*self.local_err
            
                self.global_err+=self.local_err*self.local_err
            
            print("Iteration "+iteration+": RMSE= "+np.sqrt(self.global_err/self.num_instances))
        
        test_output=self.test(self.theta,self.weights)
        print("\n=======\nNew Random Point:")

        print("class = "+test_output)
        
        
    def test(self,theta,weights):
        
        x_test=np.zeros(10)
        y_test=np.zeros(10)
        z_test=np.zeros(10)
        
        for i in range(10):
            x_test[i]=random.randrange(-10,11)
            y_test[i]=random.randrange(-10,11)
            z_test[i]=random.randrange(-10,11)
    
        output=self.calculate_output(theta,weights,x_test,y_test,z_test)
        
        return output

        
    def calculate_output(theta,weights,x,y,z):
        
        summary=x*weights[0]+y*weights[1]+z*weights[2]
        
        if summary>= theta:
            return 1
        else: return 0
        
if __name__=="__main__":
    
    percept=perceptron
    
    percept.train
