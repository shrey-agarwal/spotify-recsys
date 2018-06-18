def tester(l):
    l.append('b')
    l[4].append(3)
    
a = [1,2,3,4, [1,1], [2,2]]
tester(a)
print(a)