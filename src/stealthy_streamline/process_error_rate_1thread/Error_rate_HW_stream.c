#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>


#define SEC_LEN 128
#define SYNC_LEN 32
#define MAX_LEN 8
//const unsigned int sec[SEC_LEN] =  {2, 7, 4, 5, 6, 4, 0, 1, 3, 4, 0, 0, 4, 0, 6, 2, 3, 1, 6, 7, 6, 1, 1, 2, 6, 1, 4, 4, 2, 4, 7, 4, 3, 4, 2, 2, 0, 2, 3, 3, 6, 3, 3, 3, 3, 2, 5, 6, 3, 3, 6, 2, 4, 7, 4, 3, 0, 0, 7, 3, 4, 7, 7, 0, 3, 1, 2, 3, 4, 5, 6, 2, 0, 1, 5, 3, 3, 3, 1, 7, 6, 7, 1, 3, 7, 5, 6, 7, 6, 5, 2, 2, 4, 2, 2, 7, 3, 4, 2, 7, 1, 0, 2, 1, 2, 7, 4, 5, 2, 6, 4, 1, 5, 5, 4, 4, 3, 2, 4, 1, 7, 6, 3, 4, 0, 6, 3, 4};
const unsigned int sec[SEC_LEN] =  {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
const int threshold = 52;
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

void encode(unsigned int v_bin[3],unsigned int v){

        v_bin[0]=v/4;
        v_bin[1]=(v%4)/2;
        v_bin[2]=(v%2);
        //printf("%d %d %d %d\n", v, v_bin[0], v_bin[1],v_bin[2]);

}

int decode(unsigned int measure[8]){
    unsigned int min_res[2];
    Min(measure, 8, min_res);
    return min_res[1];
    //for(int i =0;i<8;i++){
    //    if(measure[i] < threshold) return measure[i];
    //}
    //return -1;
}




void cnt_Max(unsigned int * array, unsigned int len, unsigned int result[2]){
    //majority vote on the results in the array
    //result[0]: max #vote  => confidence
    //result[1]: the entry get max vote 
    
    unsigned int cnt[8] = {0,0,0,0, 0,0,0,0};

    for(int i=0;i<len;i++){
        if(array[i] == -1) continue;
        cnt[array[i]] ++;
    }
    Max(cnt, 8, result);
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
    
    unsigned int m_s_symbol[3], m_r_symbol[3];
    for(int i=0; i<mat_len; i++){
        encode(m_s_symbol, m_s[i%SEC_LEN]);
        encode(m_r_symbol, m_r[(i+offset)%arr_len]);
        for(int j = 0;j<3;j++){
            if(m_s_symbol[j] == 1 && m_r_symbol[j]==0){ //1->0
                cnt_1t0++;
            }
            else if(m_s_symbol[j] == 0 && m_r_symbol[j]==1){ //0->1
                cnt_0t1++;
            }
        }
        //printf("ms %d mr %d, cnt1t0 %d, cnt0t1 %d", m_s[i%SEC_LEN],m_r[(i+offset)%arr_len],cnt_1t0, cnt_0t1);
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
    //strcpy(file_name,"d");
    //sprintf(tmp_str, "%d", d);
    //strcat(file_name,tmp_str);
    //strcat(file_name,"R");
    //sprintf(tmp_str, "%d", Tr);
    //strcat(file_name,tmp_str);
    //strcat(file_name,"_");
    sprintf(tmp_str, "%d", Ts);
    strcat(file_name,tmp_str);
    sprintf(tmp_str, "_14");
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
    printf("openfile 1 %s\n", inputfile_name);
    //read = getline(&line, &len, fi);
    //printf("openfile 2 %s\n", inputfile_name);
    while ((read = getline(&line, &len, fi)) != -1) {
        //printf("%s", line);
        if(line_cnt>113){//skip first 113 lines
            unsigned int tmp_idx, tmp_measure[9];
            //unsigned long long T_monitor_real, T_monitor_next;
            printf("read sample %d\n",sample_cnt );
            sscanf(line,"%d %d %d %d %d %d %d %d %d %d %d\n", &tmp_idx, &tmp_measure[0],&tmp_measure[1],&tmp_measure[2],&tmp_measure[3],&tmp_measure[4],&tmp_measure[5],&tmp_measure[6],&tmp_measure[7], &tmp_measure[8]);
            for(int i = 0; i<8;i++){
                //fprintf(fo, "%d\t", tmp_measure[i]);
                result[sample_cnt][i]=tmp_measure[i];
            }
            //fprintf(fo, "%d\n", result[sample_cnt]);
            sample_cnt++;
            if(sample_cnt>No_test*No_itr) break;
        }
        line_cnt++;
    }
    printf("readfile Done...\n");
    

    for(int data_i=0;data_i<No_test;data_i++){
        unsigned int result_data_i[8];
        for(int j=0;j<8;j++){
                    result_data_i[j]=0;
        }  
        for(int sample_i=data_i*No_itr;sample_i<(data_i+1)*No_itr;sample_i++){
            for(int j=0;j<8;j++){
                //printf("%d %d\n", sample_i, j);
                //printf("%d\n", result[sample_i][j]);
                if(result[sample_i][j] < threshold) {
                    result_data_i[j] ++;
                }
            }  
        }

        fprintf(fo, "%d\t", data_i);
        for(int j=0;j<8;j++){
            fprintf(fo, "%d\t", result_data_i[j]);
            //printf( "%d\t", result_data_i[j]);
        }
        unsigned int result_max[2];
        Max(result_data_i, 8, result_max);
        result_data[data_i] = result_max[1];
        fprintf(fo, "%d\t %d\t", result_max[0], result_max[1]);
        fprintf(fo, "%d\t", sec[data_i%128]);
        //printf( "\n");
        fprintf(fo, "\n");


    }
    struct Error_Type error_res;
    error_res = Edit_Distance(result_data, sec, No_test, No_test, 0,fo);
    fprintf(fo, "\n  error cnt %d %d  (insertion %d, deletion %d, replacement %d, 1->0 %d, 0->1 %d)\n",  error_res.distance, No_test*3, error_res.insertion,error_res.deletion,error_res.replacement, error_res.one_to_zero,error_res.zero_to_one);

    fprintf(fo, "\n   %f\n",  (float)error_res.distance/(No_test*3));


/*

    int step=2600*len_mul;
    float period = (float)Ts/(float)Tr;
    fprintf(fo, "Period %f\n", period);
    unsigned int phase_confidents[10000];
    assert(period<10000);
    const int Threshold=43;
        
    for(int x_init=0; x_init<100000-step*period/20; x_init+=step){
        
        int last_offset=-1;
        
        for(unsigned int phase=0; phase<(int)period; phase++){
            float x=x_init+phase;
            unsigned int conf=0;
            while(x< x_init+ step*period/20-period){//get about 128*len_mul bits
                
                // process each period
                unsigned int receive_period[1024];
                for(int xx=(int)x; xx<(int)x+(int)period;xx++){
                    if (xx>=100000){
                        printf("xx out of range: x_init %d, phase %d, step %d", x_init, phase, step);
                        return NULL;
                    }
                    receive_period[xx-(int)x]=decode(result[xx]);
                }
                //printf("\n%d\n", receive_period[0]);
                unsigned int tmp_res[2];
                cnt_Max(receive_period,(int)period,tmp_res);
                fprintf(fo, "count %d :  value %d\n", tmp_res[0], tmp_res[1]);
                conf+=tmp_res[0];// confidence
                //printf("%d\n", conf);
                x+=period;
            }
            //printf("Phase %d : confidence %d\n", phase, conf);
            phase_confidents[phase]=conf;
            fprintf(fo, "Phase %d : confidence %d\n", phase, conf);
            printf("Phase %d : confidence %d\n", phase, conf);
        }
        printf("Phase conf...\n");
        unsigned int max_conf_phase[2];
        Max(phase_confidents, (int)period, max_conf_phase);
        
        unsigned int Error_phase[10000];
        int phase_range= (int)(period/6);
        if(phase_range==0) phase_range=1;
        
       
        for(int phase_idx=(int)max_conf_phase[1]-phase_range; phase_idx<(int)max_conf_phase[1]+phase_range+1; phase_idx++){// phases with max confidence
            //phase_idx might be negative here
            unsigned int receive[SEC_LEN*(MAX_LEN+2)];
            unsigned int phase = phase_idx%(int)period;
            float x=x_init+phase;
            unsigned int receive_len=0;
            while(x< x_init+ step*period/20-period){//get about 128*len_mul bits
                // process each period
                unsigned int receive_period[100000];
                for(int xx=(int)x; xx<(int)x+(int)period;xx++){
                    if (xx>=100000){
                        printf("xx out of range: x_init %d, phase %d, step %d", x_init, phase, step);
                        return NULL;
                    }
                    receive_period[xx-(int)x]=decode(result[xx]);
                }
                unsigned int tmp_res[2];
                cnt_Max(receive_period,(int)period,tmp_res);
                receive[receive_len]=tmp_res[1]; // receive string
                receive_len++;
                x+=period;
            }
            for(int i=0;i<receive_len; i+=8){
                for(int ii=i; ii<i+8;ii++) fprintf(fo, "%d", receive[ii]);
                fprintf(fo, "\t");
            }
            fprintf(fo, "\n");
            

            unsigned int error[SEC_LEN],ins[SEC_LEN],del[SEC_LEN],rep[SEC_LEN],r1t0[SEC_LEN], r0t1[SEC_LEN], min_error[2];
            struct Error_Type error_res;
            if(last_offset==-1){//first time test all offsets
                for(int offset=0; offset<SEC_LEN;offset++){
                    error_res =Edit_Distance(receive, sec, SEC_LEN, SYNC_LEN, offset,fo);
                    error[offset] =error_res.distance;
                    ins[offset] =error_res.insertion;
                    del[offset] =error_res.deletion;
                    rep[offset] =error_res.replacement;
                    r1t0[offset] = error_res.one_to_zero;
                    r0t1[offset] = error_res.zero_to_one;
                   
                }
                Min(error,SEC_LEN,min_error);
                last_offset=min_error[1];//index of the min error
            	fprintf(fo, "start at x %d\n",x_init+phase);
                fprintf(fo, "sec:\n");
                for(int i=0;i<SEC_LEN; i+=8){
                	for(int ii=i; ii<i+8;ii++) fprintf(fo, "%d", sec[(ii+last_offset)%SEC_LEN]);
                fprintf(fo, "\t");
            	}
                fprintf(fo, "\n SYNC: Phase %d %d %d when offset is %d, (insertion %d, deletion %d, replacement %d, 1->0 %d, 0->1 %d)\n", phase, receive_len, min_error[0], last_offset,ins[min_error[1]],del[min_error[1]],rep[min_error[1]], r1t0[min_error[1]],r0t1[min_error[1]]);
                error_res = Edit_Distance(receive, sec, SEC_LEN*len_mul, SEC_LEN*len_mul, last_offset,fo);
                fprintf(fo, "\n Phase %d %d %d when offset is %d, (insertion %d, deletion %d, replacement %d, 1->0 %d, 0->1 %d)\n", phase, receive_len, error_res.distance, last_offset,error_res.insertion,error_res.deletion,error_res.replacement, error_res.one_to_zero,error_res.zero_to_one);
                
            }
            else{
                for(int offset=0; offset<3;offset++){
                    error_res =Edit_Distance(receive, sec, SEC_LEN,SYNC_LEN, last_offset+offset-1,fo);
                    error[offset] =error_res.distance;
                    ins[offset] =error_res.insertion;
                    del[offset] =error_res.deletion;
                    rep[offset] =error_res.replacement;
                    r1t0[offset] = error_res.one_to_zero;
                    r0t1[offset] = error_res.zero_to_one;
                }
                Min(error,3,min_error);//min of the three offset
                last_offset=last_offset+min_error[1]-1;
            	fprintf(fo, "start at x %d\n",x_init+phase);
                fprintf(fo, "sec:\n");
                for(int i=0;i<SEC_LEN; i+=8){
                        for(int ii=i; ii<i+8;ii++) fprintf(fo, "%d", sec[(ii+last_offset)%SEC_LEN]);
                	fprintf(fo, "\t");
                }

                fprintf(fo, "\n SYNC: Phase %d %d %d when offset is %d, (insertion %d, deletion %d, replacement %d, 1->0 %d, 0->1 %d)\n", phase, receive_len, min_error[0], last_offset,ins[min_error[1]],del[min_error[1]],rep[min_error[1]],r1t0[min_error[1]],r0t1[min_error[1]]);
                error_res = Edit_Distance(receive, sec, SEC_LEN*len_mul, SEC_LEN*len_mul, last_offset,fo);
                fprintf(fo, "\n Phase %d %d %d when offset is %d, (insertion %d, deletion %d, replacement %d, 1->0 %d, 0->1 %d)\n", phase, receive_len, error_res.distance, last_offset,error_res.insertion,error_res.deletion,error_res.replacement, error_res.one_to_zero,error_res.zero_to_one);
            }
            
            
            Error_phase[phase_idx-(int)max_conf_phase[1]+phase_range]=error_res.distance;
        }
        unsigned int min_error_phase[2];
        Min(Error_phase,phase_range*2+1,min_error_phase);
        fprintf(fo,"Min error is at phase %d Error= %d\n",(min_error_phase[1]+(int)max_conf_phase[1]-phase_range)%(int)period, min_error_phase[0]);
        
    }
    
    */
    fclose(fi);
    fclose(fo);
    printf("Finish Processing %s\n", outputfile_name);
    return NULL;
}
    
int main(){
/*
    unsigned int Ts_all[10]={50000,20000,10000,5000,3200,1600};
    unsigned int Tr_all[10]={1600};
    unsigned int d_all[10]={1};

    unsigned int Ts_all[10]={320000,120000,30000,60000,90000};
    unsigned int Tr_all[10]={1600};
    unsigned int d_all[10]={1};

    unsigned int Ts_all[10]={12000};
    unsigned int Tr_all[10]={1000};
    unsigned int d_all[10]={1};
 */
    unsigned int Tr_all[10]={1600};
    unsigned int d_all[10]={1};

    unsigned int Ts_all[10]={100,10,5};
 
    unsigned int len_mul_all[4]= {1,2,4,8};
    
    pthread_t threads[100];
    struct process_file_arg arg[100];
    unsigned int file_cnt=0;
    
    for(int l_idx=0; l_idx< 1; l_idx++){
        file_cnt=0;
        for(int s_idx=0; s_idx< 1; s_idx++)
            for(int r_idx=0; r_idx< 1; r_idx++)
                for(int d_idx=0; d_idx< 1; d_idx++){
         
                    
                        assert(Ts_all[s_idx]!=0);
                        assert(Tr_all[r_idx]!=0);
                        assert(d_all[d_idx]!=0);
        
                        arg[file_cnt].d=d_all[d_idx];
                        arg[file_cnt].Ts=Ts_all[s_idx];
                        arg[file_cnt].Tr=Tr_all[r_idx];
                        arg[file_cnt].len_mul=len_mul_all[l_idx];
        
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
