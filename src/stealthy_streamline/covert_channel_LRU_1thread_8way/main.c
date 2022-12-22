/*

Covert channel using Algorithm 1 under Hyper-threaded sharing

Authors: Wenjie Xiong (wenjiex@vt.edu) 

*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <pthread.h>
#include <assert.h>
#include <time.h>
#define gettid() syscall(SYS_gettid)
#include <dlfcn.h>

#define line_size 64 // bytes
#define  way_size 64*64/8 // assuming 64 bit (8 byte) data; 64 (sets) * 64bytes (cache line size) / 8bytes (per long long)
#define  way_size_2 (64*64/8)/2

int g_stride=8;// next set

//this code uses set 48 to transfer info

char* chain_array[512*8*8]; // array to hold the pointer chasing chain  =way_size *8
unsigned long long probe_array[512*64];
unsigned int g_result[2*8*512*2*256];// to hold 100000*8 element, only use set 0-31 512= 32 set * 64 Byte(cacheline) /4byte(int)((i/512)*1024+i%512)
int message[128];
unsigned int measure_tmp[8];
unsigned int No_test = 2048;
unsigned int No_itr = 100;
unsigned int g_result_value[500000];
char** probe=NULL;
int perm[32]; 
char* error;
void *gadget_module = NULL;

int para_v=7;
int para_d=10;

char sender_mode='a';
struct readThreadParams {                                                   
 char **start;
 char **chain; 
};   


void Min(unsigned int * array, unsigned int len, unsigned int result[2]){
    // minimum value in array "array" of lenth "len"
    
    //init
    result[0] = array[0];//min value
    result[1] = 0;//index of min value
    
    for(int i=1;i<len;i++){
        if(array[i] < result[0]){
            result[0]=array[i];
            result[1]=i;
        }   
    }
}

int decode(unsigned int measure[8]){
    unsigned int min_res[2];
    Min(measure, 8, min_res);
    return min_res[1];
}

void cnt_Max(unsigned int * array, unsigned int len){
    //majority vote on the results in the array
    //result[0]: max #vote  => confidence
    //result[1]: the entry get max vote 
    
    unsigned int cnt[8] = {0,0,0,0, 0,0,0,0};

    for(int i=0;i<len;i++){
        cnt[array[i]] ++;
    }
    printf("\t");
    for(int i=0;i<8;i++){
        printf("%d\t",cnt[i]);
    }
    printf("\n");
}


void create_permutation(int size){
//  create random permutation of probe size in perm[]
	for(int i=0; i < size; i++){
		perm[i]=i;
	}

	for(int i=0; i < size; i++){
		int j = i + (rand()%(size-i));
		if(i!=j){
			int tmp=perm[i];
			perm[i]=perm[j];
			perm[j]=tmp;
		}
	
	}

	for(int i=0; i < size; i++){
		printf("% 8d", perm[i]);
	}
	printf("\n");

}


char* create_chain(int stride, int offset, char* last){
//create pointer chasing chain in probe array with stride
// offset decides which set to start
	char** start = &chain_array[perm[0]*stride + offset];
  printf("create chain stride %d, offset %d, last %p \n", stride, offset, last);
	for(int i=0; i < 8; i++){
    assert(perm[i]*stride + offset < 512*8*8);
		chain_array[perm[i]*stride + offset]= (char*) (& chain_array[perm[i+1]*stride + offset]);
        printf("%p\t",&chain_array[perm[i+1]*stride + offset]);
	}
	//chain_array[perm[6]*stride + offset] = last;
    
    printf("\n");
    void* temp=start, *temp2;
    for(int i=0; i < 8; i++){
        temp2 = (void*) *(void**)temp;
        printf("%llx, %llx\n",temp, temp2);
        temp=temp2;
    }

    unsigned long long t=0;
    printf("\n");
       asm __volatile__ (
       "movq (%%rcx),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "lfence              \n"
       "rdtsc               \n"
       : "=a" (t)
       : "c" (start));
  
     asm __volatile__ (
       "lfence              \n" 
       "rdtsc               \n"
       "movl %%eax, %%esi   \n"
       "movq (%%rcx),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "lfence              \n"
       "rdtsc               \n"
       "subl %%esi, %%eax   \n"
       : "=a" (t)
       : "c" (start)
       : "%esi", "%edx");
  printf("%d \n",t);
	return start;
}

//receiver's operation
void * test_delay(void* arg) {
  struct readThreadParams * arg_in=(struct readThreadParams *)arg;
  long long** start= arg_in->start;
  long long** chain= arg_in->chain;

  //prepare message
  
   
   int sec=0;
  for (int i = 0;i < 16;i++){
      long long tmp = *start[i];
      printf("start thread! %d %p 0x%llx 0x%llx\n", i, start[i], tmp, *(start[i]+256));
  }

  if(sender_mode>= '0' && sender_mode<= '7'){
      sec = sender_mode -'0';
      for(int i =0; i< 128; i++){
            message[i] = sec;
      }
  }
  else if(sender_mode== 'r'){
        for(int i = 0;i < 128;i++){
        message[i]= rand()%2;
      } //generate random message
  }else{ // mode == 'a'
        for(int i =0; i< 128; i++){
            message[i] = i%2;
        } 
  }
  printf("Message to be sent: ");
  for(int i = 0;i < 128;i++){
    printf("%d, ",message[i]);
  }
  printf("\n");

  //prepare measurements
  for (int i=0;i<10;i++){
      long long tmp=*start[i];
      printf("address: %d %p 0x%llx 0x%llx\n", i, start[i], tmp, *(start[i]+256));
  }
  
  printf("result array\n");
  for (int i=0;i<512;i+=16){
      printf("0x%llx \t", &g_result[i]);
  }
  printf("\n");

  unsigned long long t=0;

  unsigned int cnt=0;
  int idx=0;

  sec = message[0];


  for(cnt=0;cnt<No_test*No_itr;cnt++) 
  {

      //Init load 8 cachelines
// for (set in 0:8) {
    for(int i=0;i<para_v;i++){
       asm __volatile__ (
       "movq (%%rcx),  %%rax     \n"
       "lfence              \n"
       //"rdtsc               \n"
       : "=a" (t)
       : "c" (start[i]));
       //printf("access %p\n", start[i]);
     }

     //printf("sec %d\n", sec);
     if(cnt%No_itr == 0){
      sec = message[(cnt/No_itr)%128]*16;
    }
    //printf("sec %d\n", sec);
    asm __volatile__ (
    "movq (%%rcx),  %%rax     \n"
    "lfence              \n"
    //"rdtsc               \n"
    : "=a" (t)
    : "c" (start[sec]));
    //printf("access %p\n", start[sec]);
    idx = cnt<<3;
    idx = idx -1;
  
	  
// not needed 
  //load extra cache lines into the same set to evict the LRU line
   for(int i=para_v;i<para_d;i++){
       asm __volatile__ (
           "movq (%%rcx),  %%rax     \n"
           "lfence              \n"
           //"rdtsc               \n"
           : "=a" (t)
           : "c" (start[i]));
           //printf("access %p\n", start[i]);
   }

    for(int j=0;j<1;j++){
      idx+=1;
      //printf("preaccess %p result idx %d %d %d %d %d\n", chain[j],cnt, idx, j, (idx/512)*1024+idx%512, g_result[(idx/512)*1024+idx%512]);
    /*
     asm __volatile__ (
       "movq (%%rcx),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "lfence              \n"
       //"rdtsc               \n"
       : "=a" (t)
       : "c" (chain[j]));
  */
     asm __volatile__ (
       "lfence              \n" 
       "rdtsc               \n"
       "movl %%eax, %%esi   \n"
       "movq (%%rcx),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "movq (%%rax),  %%rax     \n"
       "lfence              \n"
       "rdtsc               \n"
       "subl %%esi, %%eax   \n"
       : "=a" (t)
       : "c" (chain[j])
       : "%esi", "%edx");
       
      g_result[(idx/(512))*1024+idx%(512)]= t;
      //printf("access %p result idx %d %d %d %d %d\n", chain[j],cnt, idx, j, (idx/512)*1024+idx%512, g_result[(idx/512)*1024+idx%512]);
    }
    //printf("test %d\n", cnt);
  }
//} // end looping for set 

  for (cnt=0;cnt<No_test*No_itr;cnt+=1) 
  {
      printf("%d\t", cnt);
      for(int j=0;j<8;j++){
        idx = cnt*8+j;
        measure_tmp[j] = g_result[(idx/(512))*1024+idx%(512)];
        printf("%d\t", g_result[(idx/(512))*1024+idx%(512)]);
      }
      idx = cnt*8;
      g_result_value[cnt] = decode(measure_tmp);
      printf("%d", g_result_value[cnt]);
      printf("\n");
  }

  cnt_Max(g_result_value, No_test);
  return start;
}

int main(int argc, char *argv[]) {


  if(argc >=2) {  
    sender_mode=argv[1][0]; 
  }
  if(argc >= 3){
        No_itr=strtol(argv[2], NULL, 10);
        printf("No_itr=%d\n",No_itr);
      }    
  //assert(No_test*No_itr <= 500000);

  char *way[32];
  srand(12345);

  for(int i=0;i<16;i++) perm[i]=i;
  create_permutation(16);

  gadget_module = dlopen("/usr/lib/libc-2.28.so",  RTLD_LAZY);
  probe= (char**)(dlsym(gadget_module,"wcpcpy"));
  printf("%s",dlerror());

  probe = probe + 1024*sizeof(void*);
    
    
  //for receiver  //set 48
  for(int i=0;i< 16;i++){// 8 way cache
    printf("%d\n", perm[i]*way_size +(way_size*3/4));
    assert(  (perm[i]*way_size +(way_size*3/4))/way_size > 32 );
    way[i]= &probe[perm[i]*way_size +(way_size*3/4)];
    printf(" %p, %d\t",way[i], *way[i]);
  }
  way[16]= &probe[way_size*5/8];
  printf(" %p, %d\n",way[16], *way[16]);
  //way[8]=&probe[10*way_size + (way_size/2) * (1-Is_sender)+(way_size/4)];
  
  
  for(int i=0;i< 10;i++){
    printf("%p\t",way[i]);
  }
  printf("\n");

  char * chain[8]; //to measure way[i] i=0 to 8
  for(int i=0;i<8;i++){
    //chain[i] = create_chain(way_size, (way_size/2)+(i+1), way[i]); //set 32
    chain[i] = create_chain(way_size, (way_size/2)+i * 64 / 8, way[i]); //set 32
  }

  printf("Pointer arrays created\n");

    unsigned int t;
  //add dummy computation to make sure the data load into L1/L2 from memory
  for(int j= 0;j <100;j++){
      for (int i = 0; i < 100000; ++i){
            t+=i;
      }
  }

  struct readThreadParams *attack_argv;
    
  attack_argv = (struct readThreadParams *)(malloc(sizeof(*attack_argv)));
  attack_argv -> start= &way;
  attack_argv -> chain= chain;
  

  test_delay(attack_argv);

  return 0;
}
