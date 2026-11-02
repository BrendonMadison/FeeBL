#ifndef READDATA_SEEN
#define READDATA_SEEN

extern "C" {
#include "memory.h"
}
#include "config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define max_buffer 512
#define max_buffer2 10000
#define max_variable_name 96


typedef struct 
{
  FILE *file;
  char buffer[max_buffer];
  char *point,*line_end;
}DATEI;


#define GETCH get_char()
#define UNGETCH(a) unget_char(a)

#define CONTENTS(a) (((a).type==T_DOUBLE)?(a).value.d:(a).value.i)

typedef struct {const char *buffer;}INPUT_HOME;

typedef enum {T_DOUBLE,T_INT,T_STRING,T_DOUBLE_ARRAY,T_INT_ARRAY,
	      T_DOUBLE_MIRROR,T_INT_MIRROR,T_DOUBLE_2,T_INT_2} VAR_TYPE;

typedef struct {VAR_TYPE type;union {double d;long int i;char *s;} value;}
VALUE;

typedef struct {
char name[max_variable_name];
VAR_TYPE type;
union {double d;
       long int i;
       double *dp;
       long int *ip;
       double dm[2];
       long int im[2];
       char *string;} value;
} VARIABLE;

typedef int INDEX;

typedef enum {START,MULT,DIV,PLUS,MINUS,END,END_INPUT,ERROR,ASSIGN,LB,RB,VAL,
	      VAR_NAME,STRING} TOKEN;



typedef struct{int number,max_number; VARIABLE *heap;}VAR_HEAP;


int get_char();
void unget_char(char a);
void store_buffer(const char *buffer);


int value_to_int(VALUE *val);

double value_to_double(VALUE *val);

void int_to_value(int i,VALUE *val);

void double_to_value(double d,VALUE *val);


void read_error(const char* error);

void read_value_unit(char *buffer);

int convert_to_value(char buffer[],VALUE *val);

TOKEN read_value(VALUE *val);

TOKEN read_string(char *buffer,int n);

void def_variable(VARIABLE *var,VAR_TYPE type);

void set_variable(VARIABLE *var,VALUE val);

void set_variable_string(VARIABLE *var,char cont[], MEMORY_ACCOUNT* m_account);

void get_variable(VARIABLE *var,VALUE *val);

void get_variable_element(VARIABLE *var,INDEX index,VALUE *val);

void set_variable_element(VARIABLE *var,INDEX index,VALUE val);

void read_skip();

void print_variable(VARIABLE *var);

VARIABLE* find_named_variable(const char* name);

void def_named_variable(const char* name,VAR_TYPE type);

void print_named_variable(char name[]);

void init_named_variable(int max_entry, MEMORY_ACCOUNT* m_account);

void set_named_variable(char name[],VALUE val);

void set_named_variable_string(char name[],char cont[], MEMORY_ACCOUNT* m_account);

int get_named_variable(char name[],VALUE *val);

TOKEN read_name0(char *buffer,int n);

TOKEN read_name(char *buffer,int n);

TOKEN read_token0(VALUE *val);

void print_token(TOKEN token);

TOKEN read_token(VALUE *val);

TOKEN read_token_name0(char buffer[],int n);

TOKEN read_token_name(char buffer[],int n);

TOKEN read_lvalue(char name[],int n);

void invert_value(VALUE *val);

void add_value(VALUE *val1,VALUE *val2,VALUE *val3);

void sub_value(VALUE *val1,VALUE *val2,VALUE *val3);

void mult_value(VALUE *val1,VALUE *val2,VALUE *val3);

void div_value(VALUE *val1,VALUE *val2,VALUE *val3);

TOKEN read_expression(VALUE*);

TOKEN read_primitive(VALUE *val);

TOKEN read_summand(VALUE *val);

TOKEN read_expression(VALUE *val);

TOKEN read_command();

void define_values(char *line, MEMORY_ACCOUNT* m_account);

void def_acc(MEMORY_ACCOUNT* m_account);

void def_param(MEMORY_ACCOUNT* m_account);

void input_values(char *buffer, MEMORY_ACCOUNT* m_account);



int file_open(DATEI *datei,char *name,char *type);

void file_close(DATEI *datei);


void file_rewind(DATEI *datei);

int file_read_line(DATEI *datei);

int file_find_word_line(DATEI *datei,char *word);

int file_find_word(DATEI *datei,char *word);

int file_skip_space_line(DATEI *datei);

int file_skip_space(DATEI *datei);

int file_next_word(DATEI *datei,char *word);

double file_get_double(DATEI *datei);

int file_read_double(DATEI *datei,double *x);

int file_read_int(DATEI *datei,int *n);

#if !HAVE_STPCPY
#define stpcpy(a,b) (strcpy(a,b)+strlen(b))
#endif

int file_read_until(DATEI *datei,char *end,char *buff,int n_max);

int file_read_braces(DATEI *datei,char *begin,char *end,char *buff,int n_max);

int file_find_braces(DATEI *datei,char *begin,char *end,char *buff,int n_max);

int file_open(DATEI *datei,char *name,char *type);

void imprimerBuffer(char* buffer);


#endif
