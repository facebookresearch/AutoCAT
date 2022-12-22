#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>


#define SEC_LEN 128
#define SYNC_LEN 32
#define MAX_LEN 8
const unsigned int sec[SEC_LEN] =  {0,0,0,0,1,0,1,0, 1,1,1,1,0,1,0,1, 0,1,1,1,0,1,0,1, 0,1,0,1,0,0,0,1, 1,0,0,0,0,0,0,0, 1,1,0,1,0,1,1,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,1,1, 0,1,1,1,1,0,1,0, 1,1,0,0,0,1,0,1, 0,0,1,1,0,1,1,0, 1,1,0,1,0,1,0,0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,0,1, 0,1,0,0,0,0,0,1, 1,0,0,1,1,0,0,1};


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
    // minimum value in array "array" of lenth "len"
    
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
    //strcpy(file_name,"d");
    //sprintf(tmp_str, "%d", d);
    //strcat(file_name,tmp_str);
    strcat(file_name,"R");
    sprintf(tmp_str, "%d", Tr);
    strcat(file_name,tmp_str);
    strcat(file_name,"_S");
    sprintf(tmp_str, "%d", Ts);
    strcat(file_name,tmp_str);
    
    printf("Processing %s\n", file_name);
    
    strcpy(inputfile_name,"data_");
    strcat(inputfile_name,file_name);
    strcat(inputfile_name,".txt");
    

    strcpy(outputfile_name,"Error_rate_");
    strcat(outputfile_name,file_name);
    strcat(outputfile_name,"_L");
    sprintf(tmp_str, "%d", len_mul);
    strcat(outputfile_name,tmp_str);
    strcat(outputfile_name,".txt");
    
    printf("Processing %s\n", outputfile_name);
    FILE * fi, *fo;
    fi=fopen(inputfile_name, "r");
    fo=fopen(outputfile_name, "w+");
    
    char *line;
    size_t len = 0;
    ssize_t read;
    unsigned int line_cnt=0;
    
    int delay[100000];
    unsigned int sample_cnt=0;
    while ((read = getline(&line, &len, fi)) != -1) {
        //printf("%s", line);
        if(line_cnt>28){//skip first 28 lines
            int tmp_idx, tmp_measure;
            //unsigned long long T_monitor_real, T_monitor_next;
            sscanf(line,"%d %d \n", &tmp_idx, &tmp_measure);
            delay[sample_cnt]=tmp_measure;
            sample_cnt++;
            if(sample_cnt>99999) break;
        }
        line_cnt++;
    }
    
    int step=2600*len_mul;
    float period = (float)Ts/(float)Tr;
    unsigned int phase_confidents[50];
    const int Threshold=52;
        
    for(int x_init=0; x_init<100000-step*period/20; x_init+=step){
        
        int last_offset=-1;
        for(unsigned int phase=0; phase<(int)period; phase++){
            float x=x_init+phase;
            unsigned int conf=0;
            while(x< x_init+ step*period/20-period){//get about 128*len_mul bits
                
                // process each period
                unsigned int delay_period=0;
                for(int xx=(int)x; xx<(int)x+(int)period;xx++){
                    if (xx>=100000){
                        printf("xx out of range: x_init %d, phase %d, step %d", x_init, phase, step);
                        return NULL;
                    }
                    if(delay[xx] >51){
                        delay_period+=Threshold;//noise
                    }
                    else{
                        delay_period+=delay[xx];
                    }
                }
                // confidence
                if(delay_period < Threshold*(int)period){
                    conf+= Threshold*(int)period-delay_period;}
                else{
                    conf+= delay_period-Threshold*(int)period;}
                
                x+=period;
            }
            phase_confidents[phase]=conf;
            fprintf(fo, "Phase %d : confidence %d\n", phase, conf);
        }
        
        unsigned int max_conf_phase[2];
        Max(phase_confidents, (int)period, max_conf_phase);
        
        unsigned int Error_phase[1000];
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
                unsigned int delay_period=0;
                for(int xx=(int)x; xx<(int)x+(int)period;xx++){
                    if(delay[xx] >55){
                        delay_period+=Threshold;//noise
                    }
                    else{
                        delay_period+=delay[xx];
                    }
                }
                // receive string
                if(delay_period < Threshold*(int)period){
                    receive[receive_len]=1;}
                else{
                    receive[receive_len]=0;}
                receive_len++;
                x+= period;
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
            fprintf(fo,"%d %d phaseerror= %d\n", phase_idx-(int)max_conf_phase[1]+phase_range, phase_idx, Error_phase[phase_idx-(int)max_conf_phase[1]+phase_range]);
        

        }
        unsigned int min_error_phase[2];
        Min(Error_phase,phase_range*2+1,min_error_phase);
        fprintf(fo,"error phase[] = \n");
        for(int i =0; i < phase_range*2+1;i++){
            fprintf(fo,"%d\t",i);
        }
        fprintf(fo,"\n");
        for(int i =0; i < phase_range*2+1;i++){
            fprintf(fo,"%d\t",Error_phase[i]);
        }
        fprintf(fo,"\n");
        fprintf(fo,"Min error is at phase %d Error= %d\n",(min_error_phase[1]+(int)max_conf_phase[1]-phase_range)%(int)period, min_error_phase[0]);
        
    }
    
    fclose(fi);
    fclose(fo);
    printf("Finish Processing %s\n", outputfile_name);
    return NULL;
}
    
int main(){

    unsigned int Ts_all[10]={40000, 100000,120000,300000,900000};
    unsigned int Tr_all[10]={1500};
    unsigned int d_all[10]={8,1,2,3,4,5,6,7,8};
/*
    unsigned int Ts_all[10]={12000};
    unsigned int Tr_all[10]={1000};
    unsigned int d_all[10]={1};
 */
 
    unsigned int len_mul_all[4]= {1,1,4,8};
    
    pthread_t threads[100];
    struct process_file_arg arg[100];
    unsigned int file_cnt=0;
    
    for(int l_idx=0; l_idx< 1; l_idx++){
        file_cnt=0;
        for(int s_idx=0; s_idx< 5; s_idx++)
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
