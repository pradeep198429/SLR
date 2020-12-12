#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# 3)Emp_data -> Build a prediction model for Churn_out_rate
# 
# Do the necessary transformations for input variables for getting better R^2 value for the model prepared.

# In[2]:


# reading a csv file using pandas library
data=pd.read_csv("emp_data.csv")


# In[3]:


data.columns


# In[4]:


plt.boxplot(data.Churn_out_rate,0,"rs",0)


# In[5]:


plt.hist(data.Churn_out_rate) #graph shows no normality


# In[6]:


plt.hist(data.Salary_hike) #shows that salry_hike is not normaly distributed


# In[7]:


plt.plot(data.Salary_hike,data.Churn_out_rate,"bo");plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")

#below graph shows that it have negative slope


# In[8]:


data.Churn_out_rate.corr(data.Salary_hike) # # correlation value between X and Y


# In[9]:


#np.corrcoef(time_prediction.Delivery_time,time_prediction.Sorting_time)


# In[10]:


# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf


# In[11]:


model1=smf.ols("Churn_out_rate~Salary_hike",data=data).fit()


# In[12]:


# For getting coefficients of the varibles used in equation
model1.params


# In[13]:


# P-values for the variables and R-squared value for prepared model
model1.summary()


# In[14]:


pred1=model1.predict(data)


# In[15]:


pred1


# In[16]:


#pred1 = model1.predict(data.iloc[:,0]) # Predicted values of churn_out_rate using the model here we are predecting 
#Churn_out_rate using Salary_hike which is zeroth column thats why i passed [:,0]


# In[17]:


#pred1=model1.predict(pd.DataFrame(data['Salary_hike'])) #predicted values of Churn_out_rate using the model
#here we r predicting Churn_out_rate using Salary_hike


# In[18]:


resid_error1=pred1-data.Churn_out_rate


# In[19]:


rmse_model1=np.sqrt(np.mean(resid_error1**2))


# In[20]:


rmse_model1


# In[21]:


model1.conf_int(0.05) # 95% confidence interval


# In[22]:


import matplotlib.pylab as plt


# In[23]:


plt.scatter(x=data['Salary_hike'],y=data['Churn_out_rate'],color='red');plt.plot(data['Salary_hike'],pred1,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate')


# In[24]:


pred1.corr(data.Salary_hike) # -1.0


# In[25]:


# Transforming x variables for accuracy
model2 = smf.ols('Churn_out_rate~np.log(Salary_hike)',data=data).fit()


# In[26]:


model2.params


# In[27]:


model2.summary()


# In[28]:


pred2=model2.predict(data)


# In[29]:


pred2


# In[30]:


resid_error_model2=pred2-data.Churn_out_rate


# In[31]:


resid_error_model2


# In[32]:


rmse_model2=np.sqrt(np.mean(resid_error_model2**2))


# In[33]:


rmse_model2


# In[34]:


model2.conf_int(0.05) # 95% confidence interval


# In[35]:


plt.scatter(x=data['Salary_hike'],y=data['Churn_out_rate'],color='red');plt.plot(data['Salary_hike'],pred2,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate')


# In[36]:


pred2.corr(data.Churn_out_rate) # 0.92


# In[37]:


# Exponential transformation
model3 = smf.ols('np.log(Churn_out_rate)~Salary_hike',data=data).fit()


# In[38]:


model3.params


# In[39]:


model3.summary()


# In[40]:


pred_log = model3.predict(pd.DataFrame(data['Salary_hike']))


# In[41]:


pred_log


# In[42]:


pred3=np.exp(pred_log)  # as we have used log(Churn_out_rata) in preparing model so we need to convert it back


# In[43]:


pred3


# In[44]:


pred3.corr(data.Churn_out_rate)


# In[45]:


plt.scatter(x=data['Salary_hike'],y=data['Churn_out_rate'],color='green');plt.plot(data.Salary_hike,np.exp(pred_log),color='blue');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate')


# In[46]:


resid_3 = pred3-data.Churn_out_rate


# In[47]:


resid_3


# In[48]:


student_resid = pd.DataFrame(model3.resid_pearson) 


# In[49]:


student_resid


# In[50]:


rmse_model3=np.sqrt(np.mean(resid_3**2))


# In[51]:


rmse_model3


# In[52]:


plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


# In[53]:


# Quadratic model
data["Salary_hike_sq"] = data.Salary_hike*data.Salary_hike


# In[54]:


model4 = smf.ols("Churn_out_rate~Salary_hike+Salary_hike_sq",data=data).fit()


# In[55]:


model4.params


# In[56]:


model4.summary()


# In[57]:


pred4 = model4.predict(data)


# In[58]:


pred4


# In[59]:


plt.scatter(data.Salary_hike,data.Churn_out_rate,c="b");plt.plot(data.Salary_hike,pred4,"r")


# In[60]:


resid_4 = pred4-data.Churn_out_rate


# In[61]:


resid_4


# In[62]:


rmse_model4=np.sqrt(np.mean(resid_4**2))


# In[63]:


rmse_model4


# In[64]:


# square root  transformation of Delivery_time
model5 = smf.ols('np.sqrt(Churn_out_rate)~Salary_hike',data=data).fit()


# In[65]:


model5.params


# In[66]:


model5.summary()


# In[67]:


pred_sq=model5.predict(pd.DataFrame(data['Salary_hike']))


# In[68]:


pred_sq


# In[69]:


pred5=np.square(pred_sq)


# In[70]:


pred5


# In[71]:


pred5.corr(data.Churn_out_rate)


# In[72]:


resid5=pred5-data.Churn_out_rate


# In[73]:


resid5


# In[74]:


rmse_model5=np.sqrt(np.mean(resid5**2))


# In[75]:


rmse_model5


# In[76]:


# reciprocal transformation of Churn_out_rate 
model6 = smf.ols('np.reciprocal(Churn_out_rate)~Salary_hike',data=data).fit()


# In[77]:


model6.params


# In[78]:


model6.summary()


# In[79]:


#log transformation of both independent and dependent variable
model7= smf.ols('np.log(Churn_out_rate)~np.log(Salary_hike)',data=data).fit()


# In[80]:


model7.params


# In[81]:


model7.summary()


# In[82]:


pred_lg=model7.predict(pd.DataFrame(data['Salary_hike']))


# In[83]:


pred_lg


# In[84]:


pred7=np.exp(pred_lg)


# In[85]:


pred7


# In[86]:


pred7.corr(data.Churn_out_rate)


# In[87]:


resid7=pred7-data.Churn_out_rate


# In[88]:


resid7


# In[89]:


rmse_model7=np.sqrt(np.mean(resid7**2))


# In[90]:


rmse_model7


# In[91]:


#                    r_sq    adj_r_sq     AIC      P_value        rmse_model
#    
#    model1          0.83    0.810        60.09    0.00            3.9975
#    model2         0.84     0.83         59.00    0.00            3.78
#    model3         0.87     0.85        -29.00   -0.002           3.54
#    model4         0.97     0.966        43.00 (x=-2.365,x^2=0.00)1.57797
#    model5         0.853    0.835        1.696   -0.008           3.784
#    model6         nan      nan           inf     nan             _____
#    model7         0.875    0.875       -30.33   0.00             3.31


#from above value if we compare then model4 is the best model among all models


# In[92]:


# so we will consider the model having highest R-Squared value which is the log transformation of both independent as well as 
#dependent variables - model4
# getting residuals of the entire data set
student_resid1 = pd.DataFrame(model4.resid_pearson)


# In[93]:


student_resid1


# In[94]:


plt.plot(model4.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


# above graph shaows that residual error will remain constatnt on inreasing the no of obeservation
# 
# 

# In[95]:


# Predicted vs actual values
plt.scatter(x=pred4,y=data.Churn_out_rate);plt.xlabel("Predicted");plt.ylabel("Actual")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




