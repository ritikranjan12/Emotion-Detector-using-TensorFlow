#include<stdio.h>

int linearsearch(int arr[],int target,int n){
    for (size_t i = 0; i < n; i++)
    {
        if(arr[i] == target){
            return i;
        }
    }
    return -1;
}

void main(){
    int arr[] = {4,7,2,89,43,12,6,-1};
    int target = 43;
    int size = sizeof(arr)/sizeof(arr[0]);

    int ans = linearsearch(arr,target,size-1);
    printf("%d",ans);
    
    
}