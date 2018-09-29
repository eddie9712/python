def pla():
      a= np.loadtxt('./pla.dat')
      b= np.zeros((400,1))
      for x in range(400):
        b[x]=a[x][4]
      w=np.zeros((1,5))
      for x in range(400):
        a[x][4]=1;
      w=a[0]
      flag=0
      while flag==0:
        flag=1
        sum=0
        for x in range(400):
         for y in range(5): 
            sum+=w[y]*a[x][y]
         if np.sign(sum)!=np.sign(b[x]):
            for k in range(5):
             w[k]=w[k]+b[x]*a[x][k]
        flag=0
        print (w[4]) 
if True: # TODO: change `False` to `True` once you finish `pla()`
    pla()
else:
    prepared.demo()
