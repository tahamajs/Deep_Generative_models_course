int fibbonaci(int n)
{
    if (n < 2)
        return 1;
    return fibbonaci(n-1) + fibbonaci(n-2);
}