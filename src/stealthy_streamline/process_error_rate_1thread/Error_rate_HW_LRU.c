#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>


#define SEC_LEN 128
#define SYNC_LEN 32
#define MAX_LEN 8
//const unsigned int sec[SEC_LEN] =  {2, 7, 4, 5, 6, 4, 0, 1, 3, 4, 0, 0, 4, 0, 6, 2, 3, 1, 6, 7, 6, 1, 1, 2, 6, 1, 4, 4, 2, 4, 7, 4, 3, 4, 2, 2, 0, 2, 3, 3, 6, 3, 3, 3, 3, 2, 5, 6, 3, 3, 6, 2, 4, 7, 4, 3, 0, 0, 7, 3, 4, 7, 7, 0, 3, 1, 2, 3, 4, 5, 6, 2, 0, 1, 5, 3, 3, 3, 1, 7, 6, 7, 1, 3, 7, 5, 6, 7, 6, 5, 2, 2, 4, 2, 2, 7, 3, 4, 2, 7, 1, 0, 2, 1, 2, 7, 4, 5, 2, 6, 4, 1, 5, 5, 4, 4, 3, 2, 4, 1, 7, 6, 3, 4, 0, 6, 3, 4};
//const unsigned int sec[SEC_LEN] = { 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1};

const unsigned int sec[SEC_LEN] = {0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1};
const int threshold = 50;
const int No_test =2048;


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


void Max(unsigned int * array, unsigned int len, unsigned int result[2]){
    // maximum value in array "array" of lenth "len"
    
    //init
    result[0] = array[0];//min value
    result[1] = 0;//index of min value
    
    for(int i=1;i<len;i++){
        if(array[i] > result[0]){
            result[0]=array[i];
            result[1]=i;
        }   
    }
}




struct Error_Type{
    unsigned int distance;
    unsigned int insertion;
    unsigned int deletion;
    unsigned int replacement;
    unsigned int one_to_zero;
    unsigned int zero_to_one;
};
struct Error_Type  Edit_Distance( const unsigned int* m_r, const unsigned int* m_s, unsigned int arr_len, unsigned int mat_len, unsigned int offset, FILE * fo){
    //This implementation compute Hamming distance
    
    unsigned int cnt_1t0=0;
    unsigned int cnt_0t1=0;
    struct Error_Type error_res;
    
    for(int i=0; i<mat_len; i++){
        if(m_s[i%SEC_LEN] == 1 && m_r[(i+offset)%arr_len]==0){ //1->0
            cnt_1t0++;
        }
        else if(m_s[i%SEC_LEN] == 0 && m_r[(i+offset)%arr_len]==1){ //0->1
            cnt_0t1++;
        }
    }
    
    error_res.distance =cnt_1t0+cnt_0t1;
    error_res.insertion =0;
    error_res.deletion =0;
    error_res.replacement =cnt_1t0+cnt_0t1;
    error_res.one_to_zero =cnt_1t0;
    error_res.zero_to_one =cnt_0t1;

    return error_res;
    
}


struct process_file_arg {                                                   
    unsigned int d;
    unsigned int Ts;
    unsigned int Tr;
    unsigned int len_mul; // nultiple of len to be processed; base line = 128;
}; 


void* process_file(void * arg){
    struct process_file_arg * arg_in=(struct process_file_arg *)arg;
    unsigned int d = arg_in->d;
    unsigned int Ts = arg_in->Ts;
    unsigned int Tr = arg_in->Tr;
    unsigned int len_mul = arg_in->len_mul;
    
    char tmp_str[10];
    char file_name[100],inputfile_name[100],outputfile_name[100];
    sprintf(tmp_str, "%d", Ts);
    strcpy(file_name,tmp_str);
    strcat(file_name,"_");
    sprintf(tmp_str, "%d", len_mul);
    strcat(file_name,tmp_str);
    
    unsigned int No_itr = Ts;
    printf("Processing %s\n", file_name);
    
    strcpy(inputfile_name,"data_");
    strcat(inputfile_name,file_name);
    strcat(inputfile_name,".txt");
    

    strcpy(outputfile_name,"Error_rate_");
    strcat(outputfile_name,file_name);
    //strcat(outputfile_name,"_L");
    //sprintf(tmp_str, "%d", len_mul);
    //strcat(outputfile_name,tmp_str);
    strcat(outputfile_name,".txt");
    
    printf("Processing %s\n", inputfile_name);
    FILE * fi, *fo;
    fi=fopen(inputfile_name, "r");
    fo=fopen(outputfile_name, "w+");
    printf("openfile %s\n", inputfile_name);
    char* line= malloc(2000 * sizeof(char));
    size_t len = 0;
    ssize_t read;
    unsigned int line_cnt=0;
    
    unsigned int result[204800][8];
    unsigned int result_data[2048];
    unsigned int sample_cnt=0;
    //printf("openfile 1 %s\n", inputfile_name);
    //read = getline(&line, &len, fi);
    //printf("openfile 2 %s\n", inputfile_name);
    while ((read = getline(&line, &len, fi)) != -1) {
        //printf("%s", line);
        if(line_cnt>145){//skip first 146 lines
            unsigned int tmp_idx, tmp_measure[9];
            //unsigned long long T_monitor_real, T_monitor_next;
            //printf("read sample %d\n",sample_cnt );
            sscanf(line,"%d %d %d %d %d %d %d %d %d %d %d\n", &tmp_idx, &tmp_measure[0],&tmp_measure[1],&tmp_measure[2],&tmp_measure[3],&tmp_measure[4],&tmp_measure[5],&tmp_measure[6],&tmp_measure[7], &tmp_measure[8]);
            for(int i = 0; i<8;i++){
                //fprintf(fo, "%d\t", tmp_measure[i]);
                result[sample_cnt][i]=tmp_measure[i];
            }
            //fprintf(fo, "\n");
            sample_cnt++;
            if(sample_cnt>No_test*No_itr) break;
        }
        line_cnt++;
    }
    //printf("readfile Done...\n");
    

    for(int data_i=0;data_i<No_test;data_i++){
        unsigned int result_data_i[8];
        for(int j=0;j<8;j++){
                    result_data_i[j]=0;
        }  
        for(int sample_i=data_i*No_itr;sample_i<(data_i+1)*No_itr;sample_i++){
            for(int j=0;j<1;j++){
                //printf("%d %d\n", sample_i, j);
                //printf("%d\n", result[sample_i][j]);
                fprintf(fo,"%d\t", result[sample_i][j]);
                if(result[sample_i][j] < threshold) {
                    result_data_i[j] ++;
                }
            } 
            fprintf(fo,"\n");  
        }

        fprintf(fo, "%d %d\t", data_i, result_data_i[0]);
        if(result_data_i[0]  > No_itr/2 )
            result_data[data_i] = 0; //more than half hit
        else
            result_data[data_i] = 1;
        fprintf(fo, "%d\t ", result_data[data_i]);
        fprintf(fo, "%d\t", sec[data_i%128]);
        //printf( "\n");
        fprintf(fo, "\n");


    }
    struct Error_Type error_res;
    error_res = Edit_Distance(result_data, sec, No_test, No_test, 0,fo);
    fprintf(fo, "\n  error cnt %d %d  (insertion %d, deletion %d, replacement %d, 1->0 %d, 0->1 %d)\n",  error_res.distance, No_test*3, error_res.insertion,error_res.deletion,error_res.replacement, error_res.one_to_zero,error_res.zero_to_one);

    fprintf(fo, "\n   %f\n",  (float)error_res.distance/(No_test));
    printf("%d %d   %f\n", Ts, len_mul, (float)error_res.distance/(No_test));
    fclose(fi);
    fclose(fo);
    //printf("Finish Processing %s\n", outputfile_name);
    return NULL;
}
    
int main(){

    unsigned int Tr_all[10]={1600};
    unsigned int d_all[10]={1};

    unsigned int Ts_all[10]={100, 50, 20, 10, 5,4,3,2,1};
 
    unsigned int try[5]= {0,1,2,3,4};
    
    pthread_t threads[100];
    struct process_file_arg arg[100];
    unsigned int file_cnt=0;
    
    for(int l_idx=0; l_idx< 5; l_idx++){
        file_cnt=0;
        for(int s_idx=0; s_idx< 9; s_idx++)
            for(int r_idx=0; r_idx< 1; r_idx++)
                for(int d_idx=0; d_idx< 1; d_idx++){
         
                    
                        assert(Ts_all[s_idx]!=0);
                        assert(Tr_all[r_idx]!=0);
                        assert(d_all[d_idx]!=0);
        
                        arg[file_cnt].d=d_all[d_idx];
                        arg[file_cnt].Ts=Ts_all[s_idx];
                        arg[file_cnt].Tr=Tr_all[r_idx];
                        arg[file_cnt].len_mul=try[l_idx];
        
                        pthread_create(&threads[file_cnt], NULL, process_file, &arg[file_cnt]);
                        file_cnt ++;
        
                }
        printf("total number of files: %d", file_cnt);
        for(int i=0; i < file_cnt; i++){
            pthread_join(threads[i], NULL);
            printf("Thread %d finished\n", i);
        }
    }
    return 0;
}
