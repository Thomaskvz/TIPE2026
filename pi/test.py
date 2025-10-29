def racineDigitale(n):
    if n<10:
        return n
    return racineDigitale(sommeDigitalle(n))



def sommeDigitalle(n):
    somme=0
    while n>0:
        somme+=n%10
        n=n//10
    return somme

print(racineDigitale(9875))
