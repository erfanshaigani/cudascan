// ONLY MODIFY THIS FILE

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// This code is going to support multiple blocked array scan
// I mean an array of size 1 to 1 << 29 elements

/*
    for sizes 1<<28 and 1<<29, you cannot copy the whole array from cpu to gpu!
    it does not fit in our gpu global memory!
    with "checkDeviceInfor.cu" I checked gpu global memory :
    Total amount of global memory: 1007747072 bytes
    which is around 1 Gbytes!
    now for an array of size 1<<27 floats we have
    1<<27 * 4 bytes / (1<<20) = 512 Mbytes!
    so for array size 1<<28, it reaches 1.024 Gbytes and 
    therefore it does not fit into the gpu mem.

    So I will make the size 1<<27 my standard size, and break bigger
    arrays into this size!
    for eample for an array with size 1<<29, I should bring 4 segments
    of size 1<<27 of the array to gpy, scan them and do some little tricks
    to be able to do that.

*/


// I performed in-place scan, meaning I scanned the array in itself, not a new array
// for the sake of mem

#include "scan2.h"
#include "gpuerrors.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

// for arrays of size : 1 to 1024 elements
__global__ void scan_kernel(float* ad, float* cd, int n)
{
    __shared__ float sd[1024]; // shared mem -> has to be fixed sized, if it was possible
    // it was better to write sd[n] but since we are not able, I allocated the max size possible in
    // this code which is 1024 
    
    //int i = bx * n + tx; // global indexing
    int j = tx; // local indexing
    int k = bx * n / 2 + tx;

    // since the input array is of size n and we have n / 2 threads, each thread 
    // will bring 2 elements.
    // actually in this code since we have only 1 block, bx = 0
    // and j and k are equal.
    sd[2 * j] = ad[2 * k];
    sd[2 * j + 1] = ad[2 * k + 1];
    //__syncthreads(); // wait until the sh mem of each block is full
    // actually we do not need this syncthread, since each thread is bringing 
    // the elements it needs in the first step of the loop
    // ex : thread 4 brings ad[8], ad[9]
    // and thread 4 needs only ad[8], ad[9] in the first step(first iteration of for) 
    int d;
    for(d = 1; d <= n / 2; d = d * 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            
        }
        __syncthreads(); // each thread has to wait here till all other threads reach here
        // then go to the next step(next iteration).
    }
    // now that the first half is done : 

    
    // this part is for inclusive scan!
    //uncomment if you want inclusive scan
    /////////////// very important!!!!!!!!!!!!!! spare variabele! each thread has it's own!
    //float spare; // for inclusive scan
    //if(j == (n / 2) - 1) // j = n/2 - 1 is the last thread, because we have n / 2 threads
    //{
    //    //float spare; // do not know why it raises an error if declared here
    //    spare = sd[n - 1]; // here spare of the last thread gets the valus only! do not expect other threads
    //    // to see any change in their spare variable!!!    
    //}
    //__syncthreads(); // VERY important! o.w it may result in wrong answer!
    //float spare; // keep it for inclusive scan
    //spare = sd[n - 1];
    


    if(j == 0) // thread number 0 or the first thread sets the last element to zero.
    {
        //spare = sd[n - 1];
        sd[n - 1] = 0;
    }
    
    // very very important note here; 
    // if you do j == anything other than 0
    // you will need __syncthreads() !
    //__syncthreads(); so we do not need sync here!!
    
    float temp; // each thread has it's own temp
    for(d = n / 2; d >= 1; d = d / 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            temp = sd[(2 * d) * (j + 1) - 1];
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            sd[d * (2 * j + 1) - 1] = temp;
        }
        __syncthreads();
    }
    // each thread copies 2 elements from sh mem to global mem
    
    
        //uncomment this part for exclusive scan
        //exclusive scan
    cd[2 * j] = sd[2 * j];
    cd[2 * j + 1] = sd[2 * j + 1];
    

    
    //inclusive scan
    // since fisrt element of the answer is 0 in belleloch algo
    //we should not keep the first element of the answer in inclusive scan.  
    //if(j == (n / 2) - 1) // again the last thread! should load back the it's spare variable 
    //// into the last element of the answer array
    //{
    //    cd[n - 2] = sd[n - 1]; 
    //    cd[n - 1] = spare; 
    //}
    //else
    //{
    //    cd[2 * j] = sd[2 * j + 1];
    //    cd[2 * j + 1] = sd[2 * j + 2];
    //}   
}

__global__ void scan_kernel_inclusive(float* ad, float* cd, int n)
{
    __shared__ float sd[1024]; // shared mem -> has to be fixed sized, if it was possible
    // it was better to write sd[n] but since we are not able, I allocated the max size possible in
    // this code which is 1024 
    
    //int i = bx * n + tx; // global indexing
    int j = tx; // local indexing
    int k = bx * n / 2 + tx;

    // since the input array is of size n and we have n / 2 threads, each thread 
    // will bring 2 elements.
    // actually in this code since we have only 1 block, bx = 0
    // and j and k are equal.
    sd[2 * j] = ad[2 * k];
    sd[2 * j + 1] = ad[2 * k + 1];
    //__syncthreads(); // wait until the sh mem of each block is full
    // actually we do not need this syncthread, since each thread is bringing 
    // the elements it needs in the first step of the loop
    // ex : thread 4 brings ad[8], ad[9]
    // and thread 4 needs only ad[8], ad[9] in the first step(first iteration of for) 
    int d;
    for(d = 1; d <= n / 2; d = d * 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            
        }
        __syncthreads(); // each thread has to wait here till all other threads reach here
        // then go to the next step(next iteration).
    }
    // now that the first half is done : 

    
    // this part is for inclusive scan!
    //uncomment if you want inclusive scan
    /////////////// very important!!!!!!!!!!!!!! spare variabele! each thread has it's own!
    float spare; // for inclusive scan
    if(j == (n / 2) - 1) // j = n/2 - 1 is the last thread, because we have n / 2 threads
    {
        //float spare; // do not know why it raises an error if declared here
        spare = sd[n - 1]; // here spare of the last thread gets the valus only! do not expect other threads
        // to see any change in their spare variable!!!    
    }
    __syncthreads(); // VERY important! o.w it may result in wrong answer!
    //float spare; // keep it for inclusive scan
    //spare = sd[n - 1];
    


    if(j == 0) // thread number 0 or the first thread sets the last element to zero.
    {
        //spare = sd[n - 1];
        sd[n - 1] = 0;
    }
    
    // very very important note here; 
    // if you do j == anything other than 0
    // you will need __syncthreads() !
    //__syncthreads(); so we do not need sync here!!
    
    float temp; // each thread has it's own temp
    for(d = n / 2; d >= 1; d = d / 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            temp = sd[(2 * d) * (j + 1) - 1];
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            sd[d * (2 * j + 1) - 1] = temp;
        }
        __syncthreads();
    }
    // each thread copies 2 elements from sh mem to global mem
    
    
        //uncomment this part for exclusive scan
        //exclusive scan
    //cd[2 * j] = sd[2 * j];
    //cd[2 * j + 1] = sd[2 * j + 1];
    

    
    //inclusive scan
    // since fisrt element of the answer is 0 in belleloch algo
    //we should not keep the first element of the answer in inclusive scan.  
    if(j == (n / 2) - 1) // again the last thread! should load back the it's spare variable 
    // into the last element of the answer array
    {
        cd[n - 2] = sd[n - 1]; 
        cd[n - 1] = spare; 
    }
    else
    {
        cd[2 * j] = sd[2 * j + 1];
        cd[2 * j + 1] = sd[2 * j + 2];
    }   
}

__global__ void scan_each_block_kernel_inc(float* ad, float* cd, float* weight, int n)
{
    // in this kernel, 1024s can be replaced by n
    // 512s -> by n/2
    // 1023s -> by n - 1
    // 511s -> by n / 2 - 1
    // I just did not do this for the readability of code.
    __shared__ float sd[1024]; // shared mem
    int i = bx * 1024 + tx; // global indexing
    int j = tx; // local indexing
    int k = bx * 512 + tx; // global indexing


    sd[2 * j] = ad[2 * k];
    sd[2 * j + 1] = ad[2 * k + 1];
    //__syncthreads(); // wait until the sh mem of each block is full
    // actually we do not need this syncthread, since each thread is bringing 
    // the elements it needs in the first step of the loop
    // ex : thread 4 of block 0 brings ad[8], ad[9]
    // and thread 4 needs only sd[8], sd[9] in the first step(first iteration of for)
    int d;
    for(d = 1; d <= n / 2; d = d * 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            
        }
        __syncthreads();
    }
    // now that the first half is done : 

    
    // this part is for inclusive scan!
    //uncomment if you want inclusive scan
    /////////////// very important!!!!!!!!!!!!!! spare variabele! each thread has it's own!
    float spare; // for inclusive scan
    if(j == 511) // last thread of each block
    {
        //float spare; // do not know why it raises an error if declared here
        spare = sd[1023]; // here spare of thread 511 gets the valus only! do not expect other threads
        // to see any change in their spare variable!!!
        weight[bx] = sd[1023]; // for the next kernel func when we want to compensate scans of blocks             
    }
    __syncthreads(); // VERY important! o.w it may result in wrong answer!
    //float spare; // keep it for inclusive scan
    //spare = sd[1023];


    if(j == 0) // thread number 0 of each block sets the last element to zero.
    {
        //spare = sd[1023];
        sd[1023] = 0;
    }
    
    // very very important note here; 
    // if you do j == anything other than 0
    // you will need __syncthreads() !
    //__syncthreads(); so we do not need sync here!!
    
    float temp; // each thread has it's own temp
    for(d = n / 2; d >= 1; d = d / 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            temp = sd[(2 * d) * (j + 1) - 1];
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            sd[d * (2 * j + 1) - 1] = temp;
        }
        __syncthreads();
    }
    // each thread copies 2 elements from sh mem to global mem
    
    /*
        //uncomment this part for exclusive scan
        //exclusive scan
    cd[2 * k] = sd[2 * j];
    cd[2 * k + 1] = sd[2 * j + 1];
    */

    
    //inclusive scan
    // since first element of the answer is 0 in belleloch algo
    //we should not keep the first element of the answer in inclusive scan.  
    if(j == 511)
    {
        cd[bx * 1024 + 1022] = sd[1023];
        cd[bx * 1024 + 1023] = spare; 
    }
    else
    {
        cd[2 * k] = sd[2 * j + 1];
        cd[2 * k + 1] = sd[2 * j + 2];
    }  

}

__global__ void scan_each_block_kernel_inc_26_27(float* ad, float* cd, float* weight, int n)
{
    // in this kernel, 1024s can be replaced by n
    // 512s -> by n/2
    // 1023s -> by n - 1
    // 511s -> by n / 2 - 1
    // I just did not do this for the readability of code.
    __shared__ float sd[1024]; // shared mem
    int i = bx * 1024 + tx; // global indexing
    int j = tx; // local indexing
    //int k = bx * 512 + tx; // global indexing
    int k = by * ((1<<25)>>1) + bx * 512 + tx;

    // each block has 1024 elements and 512 threads, so each thread brings 2 elements
    sd[2 * j] = ad[2 * k];
    sd[2 * j + 1] = ad[2 * k + 1];
    //__syncthreads(); // wait until the sh mem of each block is full
    // actually we do not need this syncthread, since each thread is bringing 
    // the elements it needs in the first step of the loop
    // ex : thread 4 of block 0 brings ad[8], ad[9]
    // and thread 4 needs only sd[8], sd[9] in the first step(first iteration of for)
    int d;
    for(d = 1; d <= n / 2; d = d * 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            
        }
        __syncthreads();
    }
    // now that the first half is done : 

    
    // this part is for inclusive scan!
    //uncomment if you want inclusive scan
    /////////////// very important!!!!!!!!!!!!!! spare variabele! each thread has it's own!
    float spare; // for inclusive scan
    if(j == 511) // last thread of each block
    {
        //float spare; // do not know why it raises an error if declared here
        spare = sd[1023]; // here spare of thread 511 gets the valus only! do not expect other threads
        // to see any change in their spare variable!!!
        weight[by * (1<<15) + bx] = sd[1023]; // for the next kernel func when we want to compensate scans of blocks             
    }
    __syncthreads(); // VERY important! o.w it may result in wrong answer!
    //float spare; // keep it for inclusive scan
    //spare = sd[1023];


    if(j == 0) // thread number 0 of each block sets the last element to zero.
    {
        //spare = sd[1023];
        sd[1023] = 0;
    }
    
    // very very important note here; 
    // if you do j == anything other than 0
    // you will need __syncthreads() !
    //__syncthreads(); so we do not need sync here!!
    
    float temp; // each thread has it's own temp
    for(d = n / 2; d >= 1; d = d / 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            temp = sd[(2 * d) * (j + 1) - 1];
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            sd[d * (2 * j + 1) - 1] = temp;
        }
        __syncthreads();
    }
    // each thread copies 2 elements from sh mem to global mem
    
    /*
        //uncomment this part for exclusive scan
        //exclusive scan
    cd[2 * k] = sd[2 * j];
    cd[2 * k + 1] = sd[2 * j + 1];
    */

    
    //inclusive scan
    // since first element of the answer is 0 in belleloch algo
    //we should not keep the first element of the answer in inclusive scan.  
    if(j == 511)
    {
        //////////////////////////////////////////////////VERYYYYYYYYYYYYYYYYYYYYYYYYYYYY
        // BY * 1<<25
        cd[by * (1<<25) + bx * 1024 + 1022] = sd[1023];
        cd[by * (1<<25) + bx * 1024 + 1023] = spare; 
    }
    else
    {
        cd[2 * k] = sd[2 * j + 1];
        cd[2 * k + 1] = sd[2 * j + 2];
    }  

}

__global__ void scan_each_block_kernel_exc(float* ad, float* cd, float* weight, int n)
{
    // in this kernel, 1024s can be replaced by n
    // 512s -> by n/2
    // 1023s -> by n - 1
    // 511s -> by n / 2 - 1
    // I just did not do this for the readability of code.
    __shared__ float sd[1024]; // shared mem
    int i = bx * 1024 + tx; // global indexing
    int j = tx; // local indexing
    int k = bx * 512 + tx; // global indexing


    sd[2 * j] = ad[2 * k];
    sd[2 * j + 1] = ad[2 * k + 1];
    //__syncthreads(); // wait until the sh mem of each block is full
    // actually we do not need this syncthread, since each thread is bringing 
    // the elements it needs in the first step of the loop
    // ex : thread 4 of block 0 brings ad[8], ad[9]
    // and thread 4 needs only sd[8], sd[9] in the first step(first iteration of for)
    int d;
    for(d = 1; d <= n / 2; d = d * 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            
        }
        __syncthreads();
    }
    // now that the first half is done : 

    
    // this part is for inclusive scan!
    //uncomment if you want inclusive scan
    /////////////// very important!!!!!!!!!!!!!! spare variabele! each thread has it's own!
   // float spare; // for inclusive scan
    if(j == 511) // last thread of each block
    {
   //     //float spare; // do not know why it raises an error if declared here
   //     spare = sd[1023]; // here spare of thread 511 gets the valus only! do not expect other threads
   //     // to see any change in their spare variable!!!
        weight[bx] = sd[1023]; // for the next kernel func when we want to compensate scans of blocks             
    }
   __syncthreads(); // VERY important! o.w it may result in wrong answer!
    //float spare; // keep it for inclusive scan
    //spare = sd[1023];


    if(j == 0) // thread number 0 of each block sets the last element to zero.
    {
        //spare = sd[1023];
        sd[1023] = 0;
    }
    
    // very very important note here; 
    // if you do j == anything other than 0
    // you will need __syncthreads() !
    //__syncthreads(); so we do not need sync here!!
    
    float temp; // each thread has it's own temp
    for(d = n / 2; d >= 1; d = d / 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            temp = sd[(2 * d) * (j + 1) - 1];
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            sd[d * (2 * j + 1) - 1] = temp;
        }
        __syncthreads();
    }
    // each thread copies 2 elements from sh mem to global mem
    
    
        //uncomment this part for exclusive scan
        //exclusive scan
    cd[2 * k] = sd[2 * j];
    cd[2 * k + 1] = sd[2 * j + 1];
    

    
    //inclusive scan
    // since first element of the answer is 0 in belleloch algo
    //we should not keep the first element of the answer in inclusive scan.  
  //  if(j == 511)
  //  {
  //      cd[bx * 1024 + 1022] = sd[1023];
  //      cd[bx * 1024 + 1023] = spare; 
  //  }
  //  else
  //  {
  //      cd[2 * k] = sd[2 * j + 1];
  //      cd[2 * k + 1] = sd[2 * j + 2];
  //  }  

}


__global__ void scan_weight_kernel(float* ad, float* cd, int n)
{
    __shared__ float sd[1024]; // shared mem
    int i = bx * n + tx; // global indexing
    int j = tx; // local indexing
    int k = bx * n / 2 + tx; // global indexing

    sd[2 * j] = ad[2 * k];
    sd[2 * j + 1] = ad[2 * k + 1];
    //__syncthreads(); // wait until the sh mem of each block is full
    
    int d;
    for(d = 1; d <= n / 2; d = d * 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            
        }
        __syncthreads();
    }
    // now that the first half is done : 

    /*
    // this part is for inclusive scan!
    //uncomment if you want inclusive scan
    /////////////// very important!!!!!!!!!!!!!! spare variabele! each thread has it's own!
    float spare; // for inclusive scan
    if(j == n / 2 - 1) // last thread of each block
    {
        //float spare; // do not know why it raises an error if declared here
        spare = sd[n - 1]; // here spare of thread 511 gets the valus only! do not expect other threads
        // to see any change in their spare variable!!!             
    }
    __syncthreads(); // VERY important! o.w it may result in wrong answer!
    //float spare; // keep it for inclusive scan
    //spare = sd[n - 1];
    */

    if(j == 0) // thread number 0 of each block sets the last element to zero.
    {
        //spare = sd[1023];
        sd[n - 1] = 0;
    }
    
    // very very important note here; 
    // if you do j == anything other than 0
    // you will need __syncthreads() !
    //__syncthreads(); so we do not need sync here!!
    
    float temp; // each thread has it's own temp
    for(d = n / 2; d >= 1; d = d / 2)
    {
        if(j < n / (2 * d))
        {
            //sd[(2 * d) * j + d - 1 + d] = sd[(2 * d) * j + d - 1] + sd[(2 * d) * j + d - 1 + d]; this is equal to the next line
            temp = sd[(2 * d) * (j + 1) - 1];
            sd[(2 * d) * (j + 1) - 1] = sd[d * (2 * j + 1) - 1] + sd[(2 * d) * (j + 1) - 1];
            sd[d * (2 * j + 1) - 1] = temp;
        }
        __syncthreads();
    }
    // each thread copies 2 elements from sh mem to global mem
    
    
        //uncomment this part for exclusive scan
        //exclusive scan
    cd[2 * k] = sd[2 * j];
    cd[2 * k + 1] = sd[2 * j + 1];
    

    /*
    //inclusive scan
    // since first element of the answer is 0 in belleloch algo
    //we should not keep the first element of the answer in inclusive scan.  
    if(j == n / 2 - 1)
    {
        cd[bx * n + n - 2] = sd[n - 1];
        cd[bx * n + n - 1] = spare; 
    }
    else
    {
        cd[2 * k] = sd[2 * j + 1];
        cd[2 * k + 1] = sd[2 * j + 2];
    }  
    */
}

__global__ void add_weight_kernel(float* cd, float* sweight)
{
   // __shared__ float sd[1024]; // shared mem
   // __shared__ float sw[1024];
    int i = bx * 1024 + tx; // global indexing
    int j = tx; // local indexing
    int k = bx * 512 + tx; // global indexing

   // sd[j] = cd[i];
   // sw[j] = sweight[j];
   // __syncthreads(); // wait until the sh mem of each block is full

   // sd[j] = sd[j] + sw[bx];
   // cd[i] = sd[j];
    cd[i] = cd[i] + sweight[bx];
}


__global__ void add_weight_kernel_26_27(float* cd, float* sweight, float lastpartans)
{
   // __shared__ float sd[1024]; // shared mem
   // __shared__ float sw[1024];
    int i = by * (1<<25) + bx * 1024 + tx; // global indexing
    int j = tx; // local indexing
    int k = bx * 512 + tx; // global indexing

   // sd[j] = cd[i];
   // sw[j] = sweight[j];
   // __syncthreads(); // wait until the sh mem of each block is full

   // sd[j] = sd[j] + sw[bx];
   // cd[i] = sd[j];
    cd[i] = cd[i] + sweight[by * (1<<15) + bx] + lastpartans;
}
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// n is the size of the array which is n = 1 << M
void gpuKernel(float* a, float* c,int n) 
{
    const int numberOfArrayElementsPerBlock = 1024;
    float* ad;
	//float* cd;
    float* weight;
    float* newweight;
    float lastpartans = 0; // this is allocated in cpu mem
    //float* sweight;
    // allocate gpu side pointers

    // if the size of the array is in range(1,2,4,...to 1024)-------------------small array---------------
    if(n <= 1024)
    {
        HANDLE_ERROR(cudaMalloc((void**)&ad, n * sizeof(float)));
        HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));
        int threads = n / 2;
        scan_kernel_inclusive <<<1, threads>>> (ad, ad, n);
        HANDLE_ERROR(cudaMemcpy(c, ad, n * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaFree(ad));
        return;
    }
    //...............................................................................................

    if(n >= (1<<28))
    {
        int iter = n / (1<<27); // number of times to perform 1<<27 sized scan serially
        int standardsize = (1<<27);
        HANDLE_ERROR(cudaMalloc((void**)&ad, standardsize * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void**)&weight, (standardsize / numberOfArrayElementsPerBlock) * sizeof(float))); // let us not over allocate it!
        HANDLE_ERROR(cudaMalloc((void**)&newweight, (standardsize / (numberOfArrayElementsPerBlock * numberOfArrayElementsPerBlock)) * sizeof(float)));

        for(int i = 0; i < iter; i++)
        {
            HANDLE_ERROR(cudaMemcpy(ad, a + i * standardsize, standardsize * sizeof(float), cudaMemcpyHostToDevice));
            int blocks = standardsize / numberOfArrayElementsPerBlock; // number of blocks
            int threads;
            threads = numberOfArrayElementsPerBlock / 2; // 512
            int division = standardsize / (1<<25);
            dim3 dimGrid(blocks / division, division);
            scan_each_block_kernel_inc_26_27 <<<dimGrid, threads>>> (ad, ad, weight, numberOfArrayElementsPerBlock); 
            ///////////----SCAN WEIGHT----////////////////////////////////////////////////////////
            int weightArraySize = blocks;
            if(weightArraySize > 1024)
            {
                blocks = weightArraySize / numberOfArrayElementsPerBlock; // because the new weight array size 
                // has changed to weightArraySize, so recalc blocks
                scan_each_block_kernel_exc <<<blocks, threads>>> (weight, weight, newweight, numberOfArrayElementsPerBlock);
                int newweightArraySize = blocks;
                threads = newweightArraySize / 2;
                scan_weight_kernel <<<1, threads>>>(newweight, newweight, newweightArraySize);
                // now add the bias 
                add_weight_kernel <<<blocks, numberOfArrayElementsPerBlock>>>(weight, newweight);
            }
            else
            {
                threads = weightArraySize / 2; // like reduce, for an array of size n I will launch n / 2 threads.
                //int blocks = n / maxThreadsPerBlock; // number of blocks
                // BLOCKS = 1 SINCE I HAVE WRITTEN THIS CODE ONLY FOR SIZE 2 to 1024
                scan_kernel <<<1, threads>>> (weight, weight, weightArraySize);
            }
            //////////////////////////////////////////////////////////////////
            blocks = standardsize / numberOfArrayElementsPerBlock; // remember the number of blocks on ad arrays
            add_weight_kernel_26_27 <<<dimGrid, numberOfArrayElementsPerBlock>>>(ad, weight, lastpartans);
            HANDLE_ERROR(cudaMemcpy(c + i * standardsize, ad, standardsize * sizeof(float), cudaMemcpyDeviceToHost));
            lastpartans = c[(i + 1) * standardsize - 1];    
        }

        HANDLE_ERROR(cudaFree(ad));
        //HANDLE_ERROR(cudaFree(cd));
        //HANDLE_ERROR(cudaFree(sweight));
        HANDLE_ERROR(cudaFree(weight));
        HANDLE_ERROR(cudaFree(newweight));
        return;
    }


    HANDLE_ERROR(cudaMalloc((void**)&ad, n * sizeof(float)));
    //HANDLE_ERROR(cudaMalloc((void**)&cd, n * sizeof(float)));

    //HANDLE_ERROR(cudaMalloc((void**)&weight, n * sizeof(float))); // overallocated it
    HANDLE_ERROR(cudaMalloc((void**)&weight, (n / numberOfArrayElementsPerBlock) * sizeof(float))); // let us not over allocate it!
    HANDLE_ERROR(cudaMalloc((void**)&newweight, (n / (numberOfArrayElementsPerBlock * numberOfArrayElementsPerBlock)) * sizeof(float)));
    // weight will have n / 1024 elements, because each block sends it's reduced ans
    // to weight and we have n / 1024 blocks.
    // copy input from cpu mem to gpu mem
    HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = n / numberOfArrayElementsPerBlock; // number of blocks
    const int maxThreadsPerBlock = 1024;
    int threads;
    if(n >= (1<<26))
    {
        threads = numberOfArrayElementsPerBlock / 2; // 512
        int division = n / (1<<25);
        dim3 dimGrid(blocks / division, division);
        scan_each_block_kernel_inc_26_27 <<<dimGrid, threads>>> (ad, ad, weight, numberOfArrayElementsPerBlock);    
    }
    else
    {
        threads = numberOfArrayElementsPerBlock / 2; // 512
        scan_each_block_kernel_inc <<<blocks, threads>>> (ad, ad, weight, numberOfArrayElementsPerBlock); // ad -> scan -> cd
        // now let us scan the weight array to add the appropriate number to each scanned input block
        // size of the weight array is equal to the number of intial blocks,
        // because each block, has sent it's reduced answer to one element in weight array 
    }
    //////////////////////////////////WEIGHT SCAN////////////////////////////////////////////////////
    int weightArraySize = blocks;
    if(weightArraySize > 1024)
    {
        blocks = weightArraySize / numberOfArrayElementsPerBlock; // because the new weight array size 
        // has changed to weightArraySize, so recalc blocks
        scan_each_block_kernel_exc <<<blocks, threads>>> (weight, weight, newweight, numberOfArrayElementsPerBlock);
        int newweightArraySize = blocks;
        threads = newweightArraySize / 2;
        scan_weight_kernel <<<1, threads>>>(newweight, newweight, newweightArraySize); // weight -> scan -> sweight
        // now add the bias 
        add_weight_kernel <<<blocks, numberOfArrayElementsPerBlock>>>(weight, newweight);
    }
    else
    {
        threads = weightArraySize / 2; // like reduce, for an array of size n I will launch n / 2 threads.
        //int blocks = n / maxThreadsPerBlock; // number of blocks
        // BLOCKS = 1 SINCE I HAVE WRITTEN THIS CODE ONLY FOR SIZE 2 to 1024
        scan_kernel <<<1, threads>>> (weight, weight, weightArraySize);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////ADD your block scanned ad to the scanned weight elements///////////
    if(n >= (1<<26))
    {
        blocks = n / numberOfArrayElementsPerBlock; // remember the number of blocks on ad arrays
        int division = n / (1<<25);
        dim3 dimGrid(blocks / division, division);
        add_weight_kernel_26_27 <<<dimGrid, numberOfArrayElementsPerBlock>>>(ad, weight,0);
    }
    else
    {
        blocks = n / numberOfArrayElementsPerBlock; // remember the number of blocks on ad arrays
        add_weight_kernel <<<blocks, numberOfArrayElementsPerBlock>>>(ad, weight);
    }
    ////////////////////////////////////////////////////////////////////////////////////
    HANDLE_ERROR(cudaMemcpy(c, ad, n * sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(ad));
    //HANDLE_ERROR(cudaFree(cd));
    //HANDLE_ERROR(cudaFree(sweight));
    HANDLE_ERROR(cudaFree(weight));
    HANDLE_ERROR(cudaFree(newweight));
}