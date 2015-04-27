#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <omp.h>
#include "CycleTimer.h"

#pragma offload_attribute(push,target(mic))
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_opadd.h>
//#include <tbb/parallel_scan.h>

#pragma offload_attribute(pop)


//double cudaFindRepeats(int *input, int length, int *output, int *output_length); 
//double cudaScan(int* start, int* end, int* resultarray);
//double cudaScanThrust(int* start, int* end, int* resultarray);
//void printCudaInfo();


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats\n"); 
    printf("  -i  --input <NAME>     Run test on given input type. Valid inputs  are: test1, random\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

void cpu_exclusive_scan(int* start, int* end, int* output)
{
#ifdef PARALLEL
    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int twod = 1; twod < N; twod*=2)
    {
        int twod1 = twod*2;
        // parallel
        for (int i = 0; i < N; i += twod1)
        {
            output[i+twod1-1] += output[i+twod-1];
        }
    }
    output[N-1] = 0;

    // downsweep phase
    for (int twod = N/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        // parallel
        for (int i = 0; i < N; i += twod1)
        {
            int tmp = output[i+twod-1];
            output[i+twod-1] = output[i+twod1-1];
            output[i+twod1-1] += tmp;
        }
    }
#endif
    int N = end - start;
    output[0] = 0;
    for (int i = 1; i < N; i++)
    {
        output[i] = output[i-1] + start[i-1];
    }
}

int cpu_find_repeats(int *start, int length, int *output){
    int count = 0, idx = 0;
    while(idx < length - 1){ 
        if(start[idx] == start[idx + 1]){
            output[count] = idx;
            count++;
        }   
        idx++;
    }   
    return count;
}

void phi_exclusive_scan(int* start, int* end, int* output)
{
    int N = end - start;
    memmove(output, start, N*sizeof(int));

    #pragma offload target(mic) inout(output:length(N))
{
    if(N<8){
        __cilkrts_end_cilk();
        __cilkrts_set_param("nworkers", "0");
    } 
    if(N<100000){
        __cilkrts_end_cilk(); 
        __cilkrts_set_param("nworkers", "1");
    } /*
    else{
       __cilkrts_end_cilk();
       __cilkrts_set_param("nworkers", "120");
    }*/

//    typedef cilk::reducer<cilk::op_add<int>> T_reducer;
	 //   T_reducer *output_reducer = new T_reducer[N];
    // upsweep phase
    for (int twod = 1; twod < N; twod*=2)
    {
        int twod1 = twod*2;
  //      int i, tid;
        // parallel
        //#pragma omp parallel num_threads(240)
        //#pragma omp parallel private(i,tid)
        { 
            //#pragma ivdep
            ////#pragma omp for 
            //tid = omp_get_thread_num();
            //#pragma simd
            cilk_for (int i = 0; i < N; i += twod1)//(i = (N/240 * tid);( i < (N/240 * (tid+1))); i += twod1)
            {
                //if(i%twod1 ==0)
                 output[i+twod1-1] += output[i+twod-1];
                  
            }
        }
        //#pragma omp barrier
    }
    output[N-1] = 0;

    // downsweep phase
    for (int twod = N/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
//        int i, tid, tmp;
        // parallel
        //#pragma omp parallel num_threads(240)
        //#pragma omp parallel private(i,tid,tmp) //simd
        {
            //#pragma ivdep
            ////#pragma omp for 
            //tid = omp_get_thread_num();
            //for (i = (N/240 * tid);(i < (N/240 * (tid+1))); i += twod1)
            //#pragma simd
            cilk_for (int i = 0; i < N; i += twod1) //cilk_for starts as many threads as needed for the for-loop parallelisation-howto use SIMD
            {
                //if(i%twod1 ==0){
                    int tmp = output[i+twod-1];
                    output[i+twod-1] = output[i+twod1-1];
                    output[i+twod1-1] += tmp;
                //}
            }

        }
        //#pragma omp barrier
    }
  }
}

int phi_find_repeats(int *start, int length, int *output){
    int count = 0, idx = 0;
    while(idx < length - 1){ 
        if(start[idx] == start[idx + 1]){
            output[count] = idx;
            count++;
        }   
        idx++;
    }   
    return count;
}

int main(int argc, char** argv)
{
    int N = 64;
    std::string test; 
    std::string input;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"test",       1, 0, 'm'},
        {"arraysize",  1, 0, 'n'},
        {"input",      1, 0, 'i'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "m:n:i:?t", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'm':
            test = optarg; 
            break;
        case 'n':
            N = atoi(optarg);
            break;
        case 'i':
            input = optarg;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    //int* inarray = new int[N];
    //int* resultarray = new int[N];
    //int* checkarray = new int[N];
    int* inarray ;
    int* resultarray ;
    int* checkarray ;
    posix_memalign((void**)&inarray, 64, N*sizeof(int));
    posix_memalign((void**)&resultarray, 64, N*sizeof(int));
    posix_memalign((void**)&checkarray, 64, N*sizeof(int));

    if (input.compare("random") == 0) {

        srand(time(NULL));

        // generate random array
        for (int i = 0; i < N; i++) {
            int val = rand() < RAND_MAX/2;
            inarray[i] = val;
            checkarray[i] = val;
        }
    } else {
        //fixed test case - you may find this useful for debugging
        for(int i = 0; i < N; i++) {
            inarray[i] = 1;
            checkarray[i] = 1;
        }  
    }


    double micTime = 50000.;
    double serialTime = 50000.;

    if (test.compare("scan") == 0) { // test exclusive scan
        // run CUDA implementation
        for (int i=0; i<4; i++) {
            double startTime = CycleTimer::currentSeconds();
            phi_exclusive_scan(inarray, inarray+N, resultarray);
            double endTime = CycleTimer::currentSeconds();
            printf("phi_one_time_time: %.3f ms\n", 1000.f * (endTime - startTime));
            micTime = std::min(micTime, endTime-startTime);
        }

        // run reference CPU implementation
        for (int i = 0; i < 3; i++) {
            double startTime = CycleTimer::currentSeconds();
            cpu_exclusive_scan(inarray, inarray+N, checkarray);
            double endTime = CycleTimer::currentSeconds();
            serialTime = std::min(serialTime, endTime - startTime);
        }

            printf("phi_time: %.3f ms\n", 1000.f * micTime);
        printf("CPU_time: %.3f ms\n", 1000.f * serialTime);
        printf("Scan: %.3fx speedup\n", serialTime / micTime);

        // validate results
        for (int i = 0; i < N; i++)
        {
            if(checkarray[i] != resultarray[i])
            {
                fprintf(stderr,
                        "Error: Device exclusive_scan outputs incorrect result."
                        " A[%d] = %d, expecting %d.\n",
                        i, resultarray[i], checkarray[i]);
                exit(1);
            }
        }
        printf("Scan outputs are correct!\n");
    } else if (test.compare("find_repeats") == 0) { // Test find_repeats
        
        // run CUDA implementation
        int parallel_size;
        for (int i=0; i<3; i++) {
            //micTime = std::min(micTime,
             //               cudaFindRepeats(inarray, N, resultarray, &cu_size));
            double startTime = CycleTimer::currentSeconds();
            parallel_size = phi_find_repeats(inarray, N, resultarray);
            double endTime = CycleTimer::currentSeconds();
            printf("phi_one_time_time: %.3f ms\n", 1000.f * (endTime - startTime));

        }

        // run reference CPU implementation
        int serial_size;
        for (int i = 0; i < 3; i++) {
            double startTime = CycleTimer::currentSeconds();
            serial_size = cpu_find_repeats(inarray, N, checkarray);
            double endTime = CycleTimer::currentSeconds();
            serialTime = std::min(serialTime, endTime - startTime);
        }

        printf("phi_time: %.3f ms\n", 1000.f * micTime);
        printf("CPU_time: %.3f ms\n", 1000.f * serialTime);
        printf("find_repeats: %.3fx speedup\n", serialTime / micTime);

        // validate results
        if(serial_size != parallel_size){
            fprintf(stderr,
                    "Error: Device find_repeats outputs incorrect size. "
                    "Expected %d, got %d.\n",
                    serial_size, parallel_size);
            exit(1);
        }
        for (int i = 0; i < serial_size; i++)
        {
            if(checkarray[i] != resultarray[i])
            {
                fprintf(stderr,
                        "Error: Device find_repeats outputs incorrect result."
                        " A[%d] = %d, expecting %d.\n",
                        i, resultarray[i], checkarray[i]);
                exit(1);
            }
        }
        printf("find_repeats outputs are correct!\n");

    } else { 
        usage(argv[0]); 
        exit(1); 
    }
    delete[] inarray;
    delete[] resultarray;
    delete[] checkarray;
    return 0;
}
