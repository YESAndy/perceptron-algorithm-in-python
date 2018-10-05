# -*- coding: utf-8 -*-
import numpy as np
import random


def train(max_itera):
    
    
    learning_rate=0.1
    num_instances=100
    theta=0
    weights=np.zeros(4)
    local_err=0
    global_err=0
    output=0
    outputs=np.zeros(num_instances)
    iteration=0
    #create train points
    
    x=np.zeros(num_instances)
    y=np.zeros(num_instances)
    z=np.zeros(num_instances)
    #class1
    for i in range(int(num_instances/2)):
        x[i]=random.randrange(4,11)
        y[i]=random.randrange(5,8)
        z[i]=random.randrange(2,6)
    #class2   
    for i in range(int(num_instances/2),num_instances):
        x[i]=random.randrange(-2,3)
        y[i]=random.randrange(-4,5)
        z[i]=random.randrange(-6,2)
     
    for i in range(4):
        weights[i]=random.randrange(0,1)
        
    
    output=calculate_output(theta,weights,x[0],y[0],z[0])
    local_err=outputs[0]-output
    
    weights[0]+=learning_rate*local_err*x[0]
    weights[1]+=learning_rate*local_err*y[0]
    weights[2]+=learning_rate*local_err*z[0]
    weights[3]+=learning_rate*local_err

    global_err+=local_err*local_err
    
    while(global_err!=0 and iteration<=max_itera):    
        iteration+=1
        
        for i in range(1,num_instances):
            
            output=calculate_output(theta,weights,x[i],y[i],z[i])
            local_err=outputs[i]-output
            
            weights[0]+=learning_rate*local_err*x[i]
            weights[1]+=learning_rate*local_err*y[i]
            weights[2]+=learning_rate*local_err*z[i]
            weights[3]+=learning_rate*local_err
        
            global_err+=local_err*local_err
        
        print("Iteration "+str(iteration)+": RMSE= "+str(np.sqrt(global_err/num_instances)))
       
    #test data
    x_test=np.zeros(10)
    y_test=np.zeros(10)
    z_test=np.zeros(10)
    
    for i in range(10):
        x_test[i]=random.randrange(-10,11)
        y_test[i]=random.randrange(-10,11)
        z_test[i]=random.randrange(-10,11)

        test_output=calculate_output(theta,weights,x_test[i],y_test[i],z_test[i])
        
        print("\n=======\nNew Random Point:")
        print("x = "+str(x_test[i])+",y = "+str(y_test[i])+ ",z = "+str(z_test[i]))
        print("class = "+str(test_output))
    
def calculate_output(theta,weights,x,y,z):
    
    summary=x*weights[0]+y*weights[1]+z*weights[2]+weights[3]
    
    if summary>= theta:
        return 1
    else: return 2
        
if __name__=="__main__":
    
    train_percept=train
    
    train_percept(100)
    
    
        
    
    
    
