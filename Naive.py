import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB #just to verify my answers


if __name__ == "__main__":
    
    #x = np.loadtxt("spamLabelled.dat")
    x = np.loadtxt(sys.argv[1])
    
    
    y = x[:,12]
    x = x[:,:12]
    
    
    true_spamClass = []
    false_spamClass=[]
    true_nonSpamClass = []
    false_nonSpamClass = []
    tot_spam = []
    tot_nonspam = []
    true_prob_feature=[]
    false_prob_feature=[]
    grand_tot = []
    
    for i in range(12):
        #sum of true and false for each feature and class
        t_spamsum=1
        t_nonspamsum=1
        f_spamsum=1
        f_nonspamsum=1
        t_sum=1
        f_sum=1
        for j in range (200):
            if (x[j,i]==1):
                t_sum=t_sum+1
            elif (x[j,i]==0):
                f_sum=f_sum+1
                
                
            if(y[j]==1 and x[j,i]==1):
                t_spamsum=t_spamsum+1
            elif (y[j]==0 and x[j,i]==1):
                t_nonspamsum=t_nonspamsum+1
            elif (y[j]==1 and x[j,i]==0):
                f_spamsum=f_spamsum+1
            elif (y[j]==0 and x[j,i]==0):
                f_nonspamsum=f_nonspamsum+1
        
        total_for_feature_spam=t_spamsum+f_spamsum
        total_for_feature_nonspam = t_nonspamsum+f_nonspamsum
        true_spamClass.append(t_spamsum)
        false_spamClass.append(f_spamsum)
        true_nonSpamClass.append(t_nonspamsum)    
        false_nonSpamClass.append(f_nonspamsum)
        tot_spam.append(total_for_feature_spam)
        tot_nonspam.append(total_for_feature_nonspam)
        true_prob_feature.append(t_sum)
        false_prob_feature.append(f_sum)
    
        
    
    i=0
    j=0
    
    prob_true_spam = []
    prob_false_spam = []
    prob_true_nonspam = []
    prob_false_nonspam = []
    for i in range (12):
        prob_true_spam.append(true_spamClass[i]/tot_spam[i])
        prob_false_spam.append(false_spamClass[i]/tot_spam[i])
        prob_true_nonspam.append(true_nonSpamClass[i]/tot_nonspam[i])
        prob_false_nonspam.append(false_nonSpamClass[i]/tot_nonspam[i])
        
    
    #x_test =np.loadtxt ("spamUnlabelled.dat")
    x_test =np.loadtxt (sys.argv[2])
    
    i = 0
    j = 0
    total_instances = tot_spam[0]+tot_nonspam[0]
    classlist = [0,1]
    P_spam = tot_spam[0]/total_instances
    P_nonspam = tot_nonspam[0]/total_instances
    
    count = 0
    print("Results:")
    for x in x_test:
        count+=1
        P_fn_givenNonSpam=1.0
        P_fn_givenSpam = 1.0
            
        for c in classlist:
            for i in range(12):
                if (c==0):
                    if(x[i]==1):
                        P_fn_givenNonSpam = P_fn_givenNonSpam*prob_true_nonspam[i]
                        
                    elif(x[i]==0):
                        P_fn_givenNonSpam = P_fn_givenNonSpam*prob_false_nonspam[i]
                        
                elif (c==1):
                    if(x[i]==1):
                        P_fn_givenSpam = P_fn_givenSpam*prob_true_spam[i]
                    elif(x[i]==0):
                        P_fn_givenSpam = P_fn_givenSpam*prob_false_spam[i]
                        
        
        
        P_nonspam_given_email = (P_fn_givenNonSpam*P_nonspam)
        P_spam_given_email = (P_fn_givenSpam*P_spam)
        
        print("-------------------")
        print("INSTANCE:",count)
        print("feature:",x)
        print("P(S|D)=",P_spam_given_email)
        print("P(S'|D)=",P_nonspam_given_email)
        if (P_nonspam_given_email>P_spam_given_email):
            print("Classification: nonspam")
            
        elif (P_nonspam_given_email<P_spam_given_email):
            print("Classification: spam")
            
    
    
    x = np.loadtxt(sys.argv[1])
    y = x[:,12]
    x = x[:,:12]
    clf = GaussianNB()
    clf.fit (x,y)
    print("\n scikit-learn's Naive Bayes model prediction:")
    
    print(clf.predict(x_test))
    
        
                
                
        
    
    
    
    
    
            
            