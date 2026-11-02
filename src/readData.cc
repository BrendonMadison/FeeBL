#include "readData.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
// Assignment and integer-expression checks occur before narrowing conversions.
long checked_integer(long double value)
{
  if (!std::isfinite(value) || value < std::numeric_limits<long>::min() ||
      value > std::numeric_limits<long>::max() || std::trunc(value) != value)
    throw std::invalid_argument("Input integer expression is outside its representable range");
  return static_cast<long>(value);
}

void validate_numeric_assignment(const VARIABLE* var, const VALUE& val)
{
  if (val.type != T_DOUBLE) return;
  if (!std::isfinite(val.value.d))
    throw std::invalid_argument(std::string("Non-finite expression for input <") + var->name + ">");
  if (var->type == T_INT || var->type == T_INT_2 || var->type == T_INT_MIRROR)
    checked_integer(val.value.d);
}

long integer_value(const VALUE& val)
{
  return val.type == T_INT ? val.value.i : checked_integer(val.value.d);
}
}  // namespace

VAR_HEAP var_heap;
INPUT_HOME input;

int get_char()
{
  int tmp;
  tmp=*(input.buffer);
  input.buffer++;
  return (int)tmp;
}

void
unget_char(char /*a*/)
{
  input.buffer--;
}

void
store_buffer(const char *buffer)
{
  input.buffer=buffer;
}

int
value_to_int(VALUE *val)
{
	if(val->type!=T_INT){
	  fprintf(stderr,"value is not an integer\n");
	  exit(1);
	}
	return (val->value).i;
}

double value_to_double(VALUE *val)
{
	if(val->type!=T_DOUBLE){
	  fprintf(stderr,"value is not a double\n");
	  exit(1);
	}
	return (val->value).d;
}

void int_to_value(int i,VALUE *val)
{
	val->type=T_INT;
	(val->value).i=i;
}

void double_to_value(double d,VALUE *val)
{
	val->type=T_DOUBLE;
	(val->value).d=d;
}

void read_error(const char* error)
{
    fprintf(stderr,"Error:\n%s\n",error);
}

void read_value_unit(char *buffer)
{
  int letter;
  letter=GETCH;
  while(isdigit(letter)){
    *(buffer++)=letter;
    letter=GETCH;
  }
  if (letter=='.'){
    *(buffer++)=letter;
    letter=GETCH;
    while(isdigit(letter)){
      *(buffer++)=letter;
      letter=GETCH;
    }
  }
  if (letter=='e'||letter=='E'){
    *(buffer++)=letter;
    letter=GETCH;
    if (letter=='-'||letter=='+'){
      *(buffer++)=letter;
      letter=GETCH;
    }
    while(isdigit(letter)){
      *(buffer++)=letter;
      letter=GETCH;
    }
  }
  *buffer=(char)0;
  UNGETCH(letter);
}

int convert_to_value(char buffer[],VALUE *val)
{
    int letter;
    char *end;
    int i=0,isdouble=0;//int i=0,n=0,isdouble=0;

    letter=buffer[0];
    if (!isdigit(letter)&&letter!='.'&&letter!='-'){
	read_error("Value is not a number");
	return 1;
    }
    while(buffer[++i]!='\0'){
	if (buffer[i]=='.'||buffer[i]=='e'||buffer[i]=='E'){
	    isdouble=1;
	}
    }
    errno = 0;
    if (isdouble){
	val->type=T_DOUBLE;
	val->value.d=strtod(buffer,&end);
        if (errno == ERANGE || !std::isfinite(val->value.d)) {
            read_error("Numeric input is non-finite or outside the representable range");
            return 1;
        }
    }
    else{
	val->type=T_INT;
	val->value.i=strtol(buffer,&end,10);
        if (errno == ERANGE) {
            read_error("Integer input is outside the representable range");
            return 1;
        }
    }
    if (*end!='\0') return 1;
    return 0;
}

TOKEN read_value(VALUE *val)
{
  //int i=0,n=1000;
  char buffer[1000];
  //int letter;
  
  read_value_unit(buffer);
  if (convert_to_value(buffer,val))
    {
      return ERROR;
    }
  return VAL;
}

TOKEN read_string(char *buffer,int n)
{
    int letter;
    int i;

    letter=GETCH;
    if (letter!=(int)'"') {
	read_error("String definition starts with wrong letter");
	return ERROR;
    }
    for (i=1;i<n;i++){
	letter=GETCH;
	if (letter=='"'){
	    *buffer='\0';
	    return STRING;
	}
	if (letter==EOF){
	    read_error("File ended while scanning of string");
	    return ERROR;
	}
	*buffer++=letter;
    }
    read_error("String too long");
    return ERROR;
}

void def_variable(VARIABLE *var,VAR_TYPE type)
{
    var->type=type;
}

void set_variable(VARIABLE *var,VALUE val)
{
    validate_numeric_assignment(var, val);
    switch(var->type){
    case T_DOUBLE:
	var->value.d=CONTENTS(val);
	break;
    case T_INT:
	var->value.i=integer_value(val);
	break;
    case T_DOUBLE_MIRROR:
	var->value.dm[0]=CONTENTS(val);
	var->value.dm[1]=-CONTENTS(val);
	break;
    case T_INT_MIRROR:
	var->value.im[0]=integer_value(val);
	var->value.im[1]=checked_integer(-static_cast<long double>(integer_value(val)));
	break;
    case T_DOUBLE_2:
	var->value.dm[0]=CONTENTS(val);
	var->value.dm[1]=CONTENTS(val);
	break;
    case T_INT_2:
	var->value.im[0]=integer_value(val);
	var->value.im[1]=integer_value(val);
	break;
    default:
      fprintf(stderr,"set_variable: case not implemented\n");
      exit(1);
    }
}

void set_variable_string(VARIABLE *var,char cont[], MEMORY_ACCOUNT* m_account)
{
  int n=0;
  if (var->type!=T_STRING){
    fprintf(stderr,"Variable is not of type string\n");
    exit(1);
  }
  
  while(cont[n++]!='\0') ;
  var->value.string=(char*)m_account->get_memory(sizeof(char)*n);

  strncpy(var->value.string,cont,sizeof(char)*n);
}

void get_variable(VARIABLE *var,VALUE *val)
{
  switch(var->type){
  case T_DOUBLE:
    val->value.d=var->value.d;
    val->type=var->type;
    break;
  case T_INT:
    val->value.i=var->value.i;
    val->type=var->type;
    break;
  case T_DOUBLE_MIRROR:
  case T_INT_MIRROR:
  case T_DOUBLE_2:
  case T_INT_2:
    fprintf(stderr,"Error: Variable <%s> not a single value\n",var->name);
    fprintf(stderr,"Error occured in get_variable\n");
    exit(1);
  case T_STRING:
    val->type=T_STRING;
    val->value.s=var->value.string;
    break;
  default:
    fprintf(stderr,"get_variable: case not implemented\n");
    exit(1);
  }
}

void get_variable_element(VARIABLE *var,INDEX index,VALUE *val)
{
  switch(var->type){
  case T_DOUBLE:
  case T_INT:
    fprintf(stderr,"Error: Variable <%s> is a single value\n",var->name);
    fprintf(stderr,"Error occured in get_variable_element\n");
    exit(1);
  case T_DOUBLE_MIRROR:
  case T_DOUBLE_2:
    val->type=T_DOUBLE;
    if (index==1){
      val->value.d=var->value.dm[0];
    }
    else{
      if (index==2){
	val->value.d=var->value.dm[1];
      }
      else{
	fprintf(stderr,"Error: Variable <%s> has only elements 1 \
and 2\n",var->name);
	fprintf(stderr,"Error occured in get_variable_element\n");
	exit(1);
      }
    }
    break;
  case T_INT_MIRROR:
  case T_INT_2:
    val->type=T_INT;
    if (index==1){
      val->value.i=var->value.im[0];
    }
    else{
      if (index==2){
	val->value.i=var->value.im[1];
      }
      else{
	fprintf(stderr,"Error: Variable <%s> has only elements 1 \
and 2\n",var->name);
	fprintf(stderr,"Error occured in get_variable_element\n");
	exit(1);
      }
    }
    break;
  default:
    fprintf(stderr,"get_variable_element: case not implemented\n");
    exit(1);
  }
}

void set_variable_element(VARIABLE *var,INDEX index,VALUE val)
{
    char msg[200];

    validate_numeric_assignment(var, val);

    switch(var->type){
    case T_DOUBLE:
    case T_INT:
        fprintf(stderr,"Error: Variable <%s> is single value\n",var->name);
	fprintf(stderr,"Error occured in set_variable_element\n");
	exit(1);
	break;
    case T_DOUBLE_MIRROR:
    case T_DOUBLE_2:
	if (index==1){
	    var->value.dm[0]=CONTENTS(val);
	}
	else{
	    if (index==2){
		var->value.dm[1]=CONTENTS(val);
	    }
	    else{
                snprintf(msg,200,"Error: Variable <%s> contains only elements 1 \
and 2\n",var->name);
                read_error(msg);
        	fprintf(stderr,"Error occured in set_variable_element\n");
		exit(1);
	    }
	}
	break;
    case T_INT_MIRROR:
    case T_INT_2:
	if (index==1){
	    var->value.im[0]=integer_value(val);
	}
	else{
	    if(index==2){
	    var->value.im[1]=integer_value(val);
	    }
	    else{
                snprintf(msg,200,"Error: Variable <%s> contains only elements 1 \
and 2\n",var->name);
                read_error(msg);
        	fprintf(stderr,"Error occured in set_variable_element\n");
		exit(1);
	    }
	}
	break;
    default:
      fprintf(stderr,"set_variable_element: case not implemented\n");
      exit(1);
    }
}

void read_skip()
{
    int letter;

    while(isspace(letter=GETCH)) ;
    if (letter==EOF) return;
    UNGETCH(letter);
}

void print_variable(VARIABLE *var)
{
    switch (var->type){
    case T_DOUBLE:
	printf("DOUBLE:%g\n",var->value.d);
	break;
    case T_INT:
	printf("INT: %ld\n",var->value.i);
	break;
    case T_DOUBLE_2:
    case T_DOUBLE_MIRROR:
	printf("DOUBLE.1: %g\n",var->value.dm[0]);
	printf("DOUBLE.2: %g\n",var->value.dm[1]);
	break;
    case T_INT_2:
    case T_INT_MIRROR:
	printf("INT.1: %ld\n",var->value.im[0]);
	printf("INT.2: %ld\n",var->value.im[1]);
	break;
    case T_STRING:
	printf("STRING: %s\n",var->value.string);
	break;
    default:
      fprintf(stderr,"print_variable: case not implemented\n");
      exit(1);
    }
}

VARIABLE* find_named_variable(const char* name)
{
    int i;
    for (i=0;i<var_heap.number;i++){
	if (!strcmp(name,var_heap.heap[i].name)){
	    return &(var_heap.heap[i]);
	}
    }
    return NULL;
}

void def_named_variable(const char* name,VAR_TYPE type)
{
    if (find_named_variable(name)==NULL){
	if (var_heap.max_number-var_heap.number>0){
	    // VARIABLE is a C POD and many historical deck keys have no explicit
	    // default assignment. Leaving the union uninitialized made their
	    // effective defaults allocator-dependent (ASan exposed INT_MIN values).
	    // Zero first, then apply the declared defaults and deck overrides.
	    memset(&var_heap.heap[var_heap.number], 0, sizeof(VARIABLE));
	    const std::size_t length =
	        std::min(strlen(name), (size_t)max_variable_name - 1U);
	    strncpy((var_heap.heap[var_heap.number]).name, name, length);
	    (var_heap.heap[var_heap.number]).name[length] = '\0';
	    (var_heap.heap[var_heap.number]).type=type;
	    var_heap.number++;
	}
	else{
	    fprintf(stderr,"Too many variables defined\n");
	    exit(1);
	}
    }
    else{
	fprintf(stderr,"Variable <%s> already exists\n",name);
	exit(1);
    }
}

void print_named_variable(char name[])
{
    VARIABLE *var;
    var=find_named_variable(name);
    if (var!=NULL){
	printf("%s\n",name);
	print_variable(var);
    }
    else{
	fprintf(stderr,"Variable <%s> does not exist\n",name);	
    }
}

void init_named_variable(int max_entry, MEMORY_ACCOUNT* m_account)
{
    var_heap.number=0;
    var_heap.max_number=max_entry;
    var_heap.heap=(VARIABLE*)m_account->get_memory(sizeof(VARIABLE)*max_entry);
}

void set_named_variable(char name[],VALUE val)
{
    char buffer[1000],*point,*end;
    int is_simple;
    INDEX index;
    VARIABLE *var;

    strncpy(buffer,name,std::min(strlen(name)+1,(size_t)999));
    point=buffer;
    while(isalnum(*point)||*point=='_') point++;
    if (*point=='\0') {
	is_simple=1;
    }
    else{
	if(*point=='.'){
	    is_simple=0;
	}
	else{
	    fprintf(stderr,"Error: Variable <%s> is only single value\n",name);
            fprintf(stderr,"Error occured in set_named_variable\n");
	    exit(1);
	}
    }
    *point='\0';
    var=find_named_variable(buffer);
    if(var==NULL) {
	fprintf(stderr,"Variable <%s> is not defined\n",buffer);
	exit(1);
    }
    if (is_simple){
	set_variable(var,val);
    }
    else{
	point++;
	index=strtol(point,&end,10);
	set_variable_element(var,index,val);
    }
}

void set_named_variable_string(char name[],char cont[], MEMORY_ACCOUNT* m_account)
{
  char buffer[1000],*point;//    char buffer[1000],*point,*end;
  //  int is_simple;
  VARIABLE *var;
  
  strncpy(buffer,name,std::min(strlen(name)+1,(size_t)999));
  point=buffer;
  while(isalnum(*point)||*point=='_') point++;
  if (*point=='\0') {
    //is_simple=1;
  }
  else{
    fprintf(stderr,"Variable <%s> is not of type string",name);
    exit(1);
  }
  *point='\0';
  var=find_named_variable(buffer);
  if(var==NULL) {
    fprintf(stderr,"Variable <%s> is not defined\n",buffer);
    exit(1);
  }
  if (var->type!=T_STRING){
    fprintf(stderr,"Variable <%s> is not of type string",name);
    exit(1);
  }
  set_variable_string(var,cont, m_account);
}

int get_named_variable(char name[],VALUE *val)
{

  char buffer[1000],*point,*end;
  int is_simple;
  INDEX index;
  VARIABLE *var;
  
  strncpy(buffer,name,std::min(strlen(name)+1,(size_t)999));
  point=buffer;
  while(isalnum(*point)||*point=='_') point++;
  if (*point=='\0') {
    is_simple=1;
  }
  else{
    if(*point=='.'){
      is_simple=0;
    }
    else{
      is_simple=1;
    }
  }
  *point='\0';
  var=find_named_variable(buffer);
    if (var==NULL) {
      /* scd temp */
      printf("Variable <%s> not found\n",buffer);
      exit(1);
      return 1;
    }
    if (is_simple){
      get_variable(var,val);
    }
    else{
      point++;
      index=strtol(point,&end,10);
      get_variable_element(var,index,val);
    }
    return 0;
}

TOKEN read_name0(char *buffer,int n)
{
  int i=0,letter;//int i=0,letter,isvar_element=0;
  
  n--;
  letter=GETCH;
  if (!isalpha(letter)){
    return ERROR;
  }
  buffer[i++]=letter;
  while((letter=GETCH)!=EOF){
    if (i==n){
      fprintf(stderr,"Name too long in read_name\n");
      exit(1);
    }
    if (isalnum(letter)||letter=='_'||letter=='.'){
      buffer[i++]=letter;
    }
    else{
      UNGETCH(letter);
      buffer[i]='\0';
      return VAR_NAME;
    }
  }
  return END_INPUT;
}

TOKEN read_name(char *buffer,int n)
{
    TOKEN token;
    token=read_name0(buffer,n);
    return token;
}

TOKEN read_token0(VALUE *val)
{
  int letter,n=1000,tmp;//int letter,i=0,n=1000,tmp;
  char buffer[1000];
  //TOKEN token;
  
  read_skip();
  letter=GETCH;
  buffer[0]='\0';
  if(letter==EOF){
    return END_INPUT;
  }
  if (isalpha(letter)){
    UNGETCH(letter);
    /*token=*/
    read_name(buffer,n);
    tmp=get_named_variable(buffer,val);
    if(tmp) return ERROR;
    return VAL;
  }
  if (isdigit(letter)||letter=='.'){
    UNGETCH(letter);
    return read_value(val);
  }
  switch(letter){
  case '{':
    return START;
  case '*':
    return MULT;
  case '/':
    return DIV;
  case '+':
    return PLUS;
  case '-':
    return MINUS;
  case '}':
    return END_INPUT;
  case ';':
    return END;
  case '=':
    return ASSIGN;
  case '"':
    UNGETCH(letter);
    return read_string(buffer,n);
  }
  return ERROR;
}

void print_token(TOKEN token)
{
  printf("<");
  switch(token){
  case START:
    printf("START");
    break;
  case ERROR:
    printf("ERROR");
    break;
  case STRING:
    printf("STRING");
    break;
  case RB:
    printf("RB");
    break;
  case LB:
    printf("LB");
    break;
  case VAL:
    printf("VAL");
    break;
  case VAR_NAME:
    printf("VAR_NAME");
    break;
  case MULT:
    printf("MULT");
    break;
  case DIV:
    printf("DIV");
    break;
  case PLUS:
    printf("PLUS");
    break;
  case MINUS:
    printf("MINUS");
    break;
  case END:
    printf("END");
    break;
  case END_INPUT:
    printf("END_INPUT");
    break;
  case ASSIGN:
    printf("ASSIGN");
    break;
  default:
    printf(" ");
  }
  printf(">");
}

TOKEN read_token(VALUE *val)
{
    TOKEN token;
    token=read_token0(val);
/*    print_token(token);*/
    return token;
}

TOKEN read_token_name0(char buffer[],int n)
{
  int letter;//int letter,i=0;
  TOKEN token;
  
  read_skip();
  letter=GETCH;
  buffer[0]='\0';
  if(letter==EOF){
    return END_INPUT;
  }
  if (isalpha(letter)){
    UNGETCH(letter);
    token=read_name(buffer,n);
    return token;
  }
  if (isdigit(letter)||letter=='.'){
    UNGETCH(letter);
    return ERROR;
  }
  switch(letter){
  case '*':
    return MULT;
  case '/':
    return DIV;
  case '+':
    return PLUS;
  case '-':
    return MINUS;
  case '}':
    return END_INPUT;
  case ';':
    return END;
  case '=':
    return ASSIGN;
  case '"':
    UNGETCH(letter);
    return read_string(buffer,n);
  }
  return ERROR;
}

TOKEN read_token_name(char buffer[],int n)
{
    TOKEN token;
    token=read_token_name0(buffer,n);
/*    printf("(1");print_token(token);printf(")2");*/
    return token;
}

TOKEN read_lvalue(char name[],int n)
{
    TOKEN token;
    token=read_token_name(name,n);
    if (token==END_INPUT) 
      { 
	return END_INPUT;
      }
    if (token!=VAR_NAME) return ERROR;
/*    printf ("<%s>\n",name);*/
    return VAR_NAME;
}

void invert_value(VALUE *val)
{
    switch(val->type){
    case T_DOUBLE:
	val->value.d=-val->value.d;
	break;
    case T_INT:
        val->value.i=checked_integer(-static_cast<long double>(val->value.i));
	break;
    default:
	break;
    }
}

void add_value(VALUE *val1,VALUE *val2,VALUE *val3)
{
    if (val1->type==T_DOUBLE||val2->type==T_DOUBLE){
	val3->value.d=CONTENTS(*val1)+CONTENTS(*val2);
	val3->type=T_DOUBLE;
    }
    else{
	val3->value.i=checked_integer(static_cast<long double>(val1->value.i)+val2->value.i);
	val3->type=T_INT;
    }
}

void sub_value(VALUE *val1,VALUE *val2,VALUE *val3)
{
    if (val1->type==T_DOUBLE||val2->type==T_DOUBLE){
	val3->value.d=CONTENTS(*val1)-CONTENTS(*val2);
	val3->type=T_DOUBLE;
    }
    else{
	val3->value.i=checked_integer(static_cast<long double>(val1->value.i)-val2->value.i);
	val3->type=T_INT;
    }
}

void mult_value(VALUE *val1,VALUE *val2,VALUE *val3)
{
    if (val1->type==T_DOUBLE||val2->type==T_DOUBLE){
	val3->value.d=CONTENTS(*val1)*CONTENTS(*val2);
	val3->type=T_DOUBLE;
    }
    else{
	val3->value.i=checked_integer(static_cast<long double>(val1->value.i)*val2->value.i);
	val3->type=T_INT;
    }
}

void div_value(VALUE *val1,VALUE *val2,VALUE *val3)
{
    if (CONTENTS(*val2) == 0)
      throw std::invalid_argument("Division by zero in input expression");
    if (val1->type==T_DOUBLE||val2->type==T_DOUBLE){
	val3->value.d=CONTENTS(*val1)/CONTENTS(*val2);
	val3->type=T_DOUBLE;
    }
    else{
	// Integer division preserves the card language's truncation convention.
	val3->value.i=checked_integer(std::trunc(static_cast<long double>(val1->value.i)/val2->value.i));
	val3->type=T_INT;
    }
}

TOKEN read_primitive(VALUE *val)
{
    TOKEN token;
    VALUE val1;
    token=read_token(val);
    switch(token){
    case VAL:
	break;
    case MINUS:
	token=read_primitive(val);
	invert_value(val);
	return token;
    case LB:
	token=read_expression(val);
	if (token!=VAL) return ERROR;
	if ((token=read_token(&val1))!=RB) return ERROR;
	return token;
    default:
	return ERROR;
    }
    return read_token(&val1);
}

TOKEN read_summand(VALUE *val)
{
    TOKEN token;
    VALUE val1;
    token=read_primitive(val);
    while((token==MULT)||(token==DIV)){
	switch(token){
	case MULT:
	    token=read_primitive(&val1);
	    mult_value(val,&val1,val);
	    break;
	case DIV:
	    token=read_primitive(&val1);
	    div_value(val,&val1,val);
	    break;
	// unnecessary but default statement to prevent compiler warnings
	default:
	    break;
	}
    }
    return token;
}

TOKEN read_expression(VALUE *val)
{
    VALUE val1;
    TOKEN token;
    token=read_summand(val);
    if ((token==MINUS)||(token==PLUS)){
	if (token==MINUS){
	    token=read_expression(&val1);
	    if (token==ERROR) return ERROR;
	    sub_value(val,&val1,val);
	}
	else{
	    token=read_expression(&val1);
	    if (token==ERROR) return ERROR;
	    add_value(val,&val1,val);
	}
    }
    return token;
}

TOKEN read_command(MEMORY_ACCOUNT* m_account)
{
  char lval[1000],rval[1000];
  VALUE val;
  VARIABLE *tmp;
  TOKEN token;
  
  if ((token=read_lvalue(lval,1000))==END_INPUT)
    { 
      return END_INPUT;
    }
  if (token!=VAR_NAME)
    {
      return ERROR;
    }
  if (read_token(&val)!=ASSIGN)
    {
      return ERROR;
    }
  tmp=find_named_variable(lval);
  if (tmp!=NULL){
    if (tmp->type==T_STRING){
      read_string(rval,1000);
      set_named_variable_string(lval,rval, m_account);
      if (read_token(&val)!=END) return ERROR;
      return ASSIGN;
    }
  }
  if (read_expression(&val)!=END){
    return ERROR;
  }
  set_named_variable(lval,val);
  return ASSIGN;
}

void define_values(const char *line, MEMORY_ACCOUNT* m_account)
{
  TOKEN token;
  VALUE value;
  store_buffer(line);
  if ((token=read_token(&value))!=START){
	printf("error: no proper start\n");
	exit(1);
  }
  while((token=read_command(m_account))!=END_INPUT) {
    //	print_token(token);
    if (token==ERROR) {
      printf("error in define_values\n");
      exit(1);
    }
  }
}

void def_acc(MEMORY_ACCOUNT* m_account)
{
    def_named_variable("particles",T_DOUBLE_2);
    def_named_variable("energy",T_DOUBLE_2);
    def_named_variable("charge_sign",T_DOUBLE);
    def_named_variable("f_rep",T_DOUBLE);
    def_named_variable("n_b",T_INT);
    def_named_variable("sigma_x",T_DOUBLE_2);
    def_named_variable("sigma_y",T_DOUBLE_2);
    def_named_variable("sigma_z",T_DOUBLE_2);
    def_named_variable("emitt_x",T_DOUBLE_2);
    def_named_variable("emitt_y",T_DOUBLE_2);
    def_named_variable("beta_x",T_DOUBLE_2);
    def_named_variable("beta_y",T_DOUBLE_2);
    def_named_variable("offset_x",T_DOUBLE_MIRROR);
    def_named_variable("offset_y",T_DOUBLE_MIRROR);
    def_named_variable("offset_z",T_DOUBLE_MIRROR);
    def_named_variable("waist_x",T_DOUBLE_2);
    def_named_variable("waist_y",T_DOUBLE_2);
    def_named_variable("xi_x",T_DOUBLE_2);
    def_named_variable("xi_y",T_DOUBLE_2);
    def_named_variable("couple_xy",T_DOUBLE_2);
    def_named_variable("angle_phi",T_DOUBLE_2);
    def_named_variable("angle_x",T_DOUBLE_2);
    def_named_variable("angle_y",T_DOUBLE_2);
    def_named_variable("dist_x",T_INT_2);
    //    def_named_variable("dist_y",T_INT_2);
    def_named_variable("dist_z",T_INT_2);
    def_named_variable("trav_focus",T_INT_2);
    def_named_variable("espread",T_DOUBLE_2);
    def_named_variable("which_espread",T_INT_2);
/*scd 2.11.1998 DF*/
    def_named_variable("scale_step",T_DOUBLE_2);

    def_named_variable("polar_x",T_DOUBLE_2);
    def_named_variable("polar_y",T_DOUBLE_2);
    def_named_variable("polar_z",T_DOUBLE_2);

#ifdef TWOBEAM
    def_named_variable("twobeam",T_INT);
    def_named_variable("particles_2",T_DOUBLE_2);
    def_named_variable("energy_2",T_DOUBLE_2);
    //    def_named_variable("charge_sign_2",T_DOUBLE);
    def_named_variable("sigma_x_2",T_DOUBLE_2);
    def_named_variable("sigma_y_2",T_DOUBLE_2);
    def_named_variable("sigma_z_2",T_DOUBLE_2);
    def_named_variable("emitt_x_2",T_DOUBLE_2);
    def_named_variable("emitt_y_2",T_DOUBLE_2);
    def_named_variable("beta_x_2",T_DOUBLE_2);
    def_named_variable("beta_y_2",T_DOUBLE_2);
    def_named_variable("offset_x_2",T_DOUBLE_MIRROR);
    def_named_variable("offset_y_2",T_DOUBLE_MIRROR);
    def_named_variable("offset_z_2",T_DOUBLE_MIRROR);
    def_named_variable("waist_x_2",T_DOUBLE_2);
    def_named_variable("waist_y_2",T_DOUBLE_2);
    def_named_variable("angle_phi_2",T_DOUBLE_2);
    def_named_variable("angle_x_2",T_DOUBLE_2);
    def_named_variable("angle_y_2",T_DOUBLE_2);
    def_named_variable("dist_x_2",T_INT_2);
    def_named_variable("dist_y_2",T_INT_2);
    def_named_variable("dist_z_2",T_INT_2);
    def_named_variable("trav_focus_2",T_INT_2);
    def_named_variable("espread_2",T_DOUBLE_2);
    def_named_variable("which_espread_2",T_INT_2);
#endif

    define_values("{xi_x=0.0;xi_y=0.0;}", m_account);
    define_values("{offset_x=0.0;offset_y=0.0;offset_z=0.0;}", m_account);
    define_values("{waist_x=0.0;waist_y=0.0;couple_xy=0.0;}", m_account);
    define_values("{angle_phi=0.0;angle_x=0.0;angle_y=0.0;}", m_account);
    define_values("{beta_x=-1.0;beta_y=-1.0;}", m_account);
    define_values("{sigma_x=-1.0;sigma_y=-1.0;}", m_account);
    define_values("{emitt_x=-1.0;emitt_y=-1.0;}", m_account);
    define_values("{charge_sign=-1;espread=1e-3;which_espread=1;}", m_account);
/*scd 2.11.1998 DF*/
    define_values("{scale_step=1.0;}", m_account);
    define_values("{polar_x=0.0;polar_y=0.0;polar_z=1.0;}", m_account);

#ifdef TWOBEAM
    define_values("{twobeam=0;}", m_account);
    define_values("{offset_x_2=0.0;offset_y_2=0.0;offset_z_2=0.0;}", m_account);
    define_values("{waist_x_2=0.0;waist_y_2=0.0;}", m_account);
    define_values("{angle_phi_2=0.0;angle_x_2=0.0;angle_y_2=0.0;}", m_account);
    define_values("{beta_x_2=-1.0;beta_y_2=-1.0;}", m_account);
    define_values("{sigma_x_2=-1.0;sigma_y_2=-1.0;}", m_account);
    define_values("{emitt_x_2=-1.0;emitt_y_2=-1.0;}", m_account);
    //    define_values("{charge_sign_2=-1;}", m_account);
    define_values("{espread_2=1e-3;which_espread_2=1;}", m_account);
#endif
}

void def_param(MEMORY_ACCOUNT* m_account)
{
  //  def_named_variable("silent",T_INT);
  def_named_variable("n_x",T_INT);
  def_named_variable("n_y",T_INT);
  def_named_variable("n_z",T_INT);
  def_named_variable("n_t",T_INT);
  def_named_variable("n_m",T_INT_2);
  def_named_variable("cut_x",T_DOUBLE);
  def_named_variable("cut_y",T_DOUBLE);
  def_named_variable("cut_z",T_DOUBLE);

  def_named_variable("cut_x_factor",T_DOUBLE);
  def_named_variable("cut_y_factor",T_DOUBLE);
  def_named_variable("cut_z_factor",T_DOUBLE);



  def_named_variable("integration_method",T_INT);
  def_named_variable("electron_distribution_rho",T_INT);
  def_named_variable("force_symmetric",T_INT);
  def_named_variable("charge_symmetric",T_INT);
  def_named_variable("do_bds_spin_rotation",T_INT);
  def_named_variable("do_size_log",T_INT);

  def_named_variable("rndm_seed",T_INT);
  def_named_variable("rndm_load",T_INT);
  def_named_variable("rndm_save",T_INT);

  def_named_variable("compute_backend",T_STRING);
  def_named_variable("collision_schedule",T_STRING);
  def_named_variable("wavefront_parallel_physics",T_INT);
  def_named_variable("sparse_cells",T_INT);
  def_named_variable("wavefront_field_solver",T_STRING);
  def_named_variable("wavefront_workspace_max_bytes",T_DOUBLE);
  def_named_variable("sparse_direct_interaction_limit",T_INT);
  def_named_variable("cpu_threads",T_INT);
  def_named_variable("cpu_variant",T_STRING);
  def_named_variable("cpu_primary_move",T_INT);
  def_named_variable("gpu_variant",T_STRING);
  def_named_variable("gpu_device",T_INT);
  def_named_variable("gpu_min_particles",T_INT);
  def_named_variable("gpu_mode",T_INT);
  def_named_variable("gpu2_scope",T_STRING);
  def_named_variable("gpu2_deposition",T_STRING);
  def_named_variable("gpu2_fft",T_STRING);
  def_named_variable("gpu2_streams",T_INT);
  def_named_variable("gpu2_graphs",T_STRING);
  def_named_variable("gpu2_fusion",T_STRING);
  def_named_variable("gpu2_batch_slices",T_INT);
  def_named_variable("gpu2_pinned_io",T_STRING);
  def_named_variable("gpu2_host_threads",T_INT);
  def_named_variable("gpu2_primary_mover",T_STRING);
  def_named_variable("gpu2_memory_fraction",T_DOUBLE);
  def_named_variable("gpu2_profile",T_STRING);
  def_named_variable("rng_backend",T_STRING);
  def_named_variable("reduction_mode",T_STRING);
  def_named_variable("allow_backend_fallback",T_INT);
  def_named_variable("engine",T_STRING);
  def_named_variable("algorithm",T_STRING);
  def_named_variable("precision",T_STRING);
  def_named_variable("modern_reduction",T_STRING);
  def_named_variable("modern_batch_width",T_INT);
  def_named_variable("modern_cpu_workers",T_INT);
  def_named_variable("modern_cpu_runtime",T_STRING);
  def_named_variable("modern_cpu_affinity",T_STRING);
  def_named_variable("modern_gpu_streams",T_INT);
  def_named_variable("modern_gpu_graphs",T_STRING);
  def_named_variable("modern_trace",T_STRING);

  def_named_variable("adaptive_grid_mode",T_INT);
  def_named_variable("adaptive_grid_scope",T_STRING);
  def_named_variable("adaptive_grid_size_classes_x",T_STRING);
  def_named_variable("adaptive_grid_size_classes_y",T_STRING);
  def_named_variable("adaptive_grid_min_nx",T_INT);
  def_named_variable("adaptive_grid_min_ny",T_INT);
  def_named_variable("adaptive_grid_max_nx",T_INT);
  def_named_variable("adaptive_grid_max_ny",T_INT);
  def_named_variable("adaptive_grid_extra_guard_cells",T_INT);
  def_named_variable("adaptive_grid_allow_full_window",T_INT);
  def_named_variable("adaptive_grid_max_cells",T_DOUBLE);
  def_named_variable("adaptive_grid_max_bytes",T_DOUBLE);
  def_named_variable("adaptive_grid_schedule_save",T_STRING);
  def_named_variable("adaptive_grid_schedule_load",T_STRING);
  def_named_variable("adaptive_grid_profile",T_STRING);
  def_named_variable("adaptive_grid_pilot_diagnostics",T_INT);

  def_named_variable("adaptive_grid_field_spacing_classes",T_STRING);
  def_named_variable("adaptive_grid_field_operator",T_STRING);
  def_named_variable("adaptive_grid_max_refinement_factor",T_DOUBLE);
  def_named_variable("adaptive_grid_anchor_x",T_DOUBLE);
  def_named_variable("adaptive_grid_anchor_y",T_DOUBLE);
  def_named_variable("adaptive_grid_monitor",T_STRING);
  def_named_variable("adaptive_grid_error_target",T_DOUBLE);
  def_named_variable("adaptive_grid_error_norm_policy",T_STRING);
  def_named_variable("adaptive_grid_error_budget_sha256",T_STRING);
  def_named_variable("adaptive_grid_error_quantile",T_DOUBLE);
  def_named_variable("adaptive_grid_error_catastrophe_limit",T_DOUBLE);
  def_named_variable("adaptive_grid_error_reference_field",T_DOUBLE);
  def_named_variable("adaptive_grid_error_absolute_floor_fraction",T_DOUBLE);
  def_named_variable("adaptive_grid_error_catastrophe_absolute_fraction",T_DOUBLE);
  def_named_variable("adaptive_grid_error_exposure_limit",T_DOUBLE);
  def_named_variable("adaptive_grid_noise_aware_stop",T_INT);
  def_named_variable("adaptive_grid_source_model",T_STRING);
  def_named_variable("adaptive_grid_source_reference_nx",T_DOUBLE);
  def_named_variable("adaptive_grid_source_reference_ny",T_DOUBLE);
  def_named_variable("adaptive_grid_source_width_x",T_DOUBLE);
  def_named_variable("adaptive_grid_source_width_y",T_DOUBLE);
  def_named_variable("adaptive_grid_source_error_target",T_DOUBLE);
  def_named_variable("adaptive_grid_solver_error_target",T_DOUBLE);
  def_named_variable("adaptive_grid_lumi_error_target",T_DOUBLE);
  def_named_variable("adaptive_grid_lumi_floor",T_DOUBLE);
  def_named_variable("adaptive_grid_tau_curvature",T_DOUBLE);
  def_named_variable("adaptive_grid_tau_field_gradient",T_DOUBLE);
  def_named_variable("adaptive_grid_tau_richardson",T_DOUBLE);
  def_named_variable("adaptive_grid_tau_tail",T_DOUBLE);
  def_named_variable("adaptive_grid_encounter_tail_fraction",T_DOUBLE);
  def_named_variable("adaptive_grid_rho_floor",T_DOUBLE);
  def_named_variable("adaptive_grid_field_floor",T_DOUBLE);
  def_named_variable("adaptive_grid_indicator_smoothing_passes",T_INT);
  def_named_variable("adaptive_grid_indicator_reduction",T_STRING);
  def_named_variable("adaptive_grid_refine_threshold",T_DOUBLE);
  def_named_variable("adaptive_grid_coarsen_threshold",T_DOUBLE);
  def_named_variable("adaptive_grid_shrink_dwell",T_INT);
  def_named_variable("adaptive_grid_update_interval",T_INT);
  def_named_variable("adaptive_grid_richardson_interval",T_INT);
  def_named_variable("adaptive_grid_richardson_order_source",T_STRING);
  def_named_variable("adaptive_grid_richardson_order_table",T_STRING);
  def_named_variable("adaptive_grid_min_neff",T_DOUBLE);
  def_named_variable("adaptive_grid_fail_on_unmet_error",T_INT);

  def_named_variable("adaptive_macroparticle_mode",T_INT);
  def_named_variable("adaptive_macroparticle_domain",T_STRING);
  def_named_variable("adaptive_macroparticle_source_evolution",T_STRING);
  def_named_variable("adaptive_macroparticle_decision_scope",T_STRING);
  def_named_variable("adaptive_macroparticle_commit_scope",T_STRING);
  def_named_variable("adaptive_macroparticle_strategy",T_STRING);
  def_named_variable("adaptive_macroparticle_split",T_STRING);
  def_named_variable("adaptive_macroparticle_merge",T_STRING);
  def_named_variable("adaptive_macroparticle_root_source",T_STRING);
  def_named_variable("adaptive_macroparticle_root_seed",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_scramble_id",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_error_target",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_resample_bias_target",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_min_root_support",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_min_clone_support",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_statistical_cohorts",T_INT);
  def_named_variable("adaptive_macroparticle_refine_threshold",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_coarsen_threshold",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_update_interval",T_INT);
  def_named_variable("adaptive_macroparticle_cooldown",T_INT);
  def_named_variable("adaptive_macroparticle_dwell",T_INT);
  def_named_variable("adaptive_macroparticle_max_change_factor",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_max_generation",T_INT);
  def_named_variable("adaptive_macroparticle_max_count",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_max_bytes",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_max_weight_ratio",T_DOUBLE);
  def_named_variable("adaptive_macroparticle_policy_save",T_STRING);
  def_named_variable("adaptive_macroparticle_policy_load",T_STRING);
  def_named_variable("adaptive_macroparticle_trace_save",T_STRING);
  def_named_variable("adaptive_macroparticle_trace_load",T_STRING);
  def_named_variable("adaptive_macroparticle_profile",T_STRING);
  def_named_variable("adaptive_macroparticle_fail_on_unmet_error",T_INT);
  def_named_variable("adaptive_macroparticle_determinism",T_STRING);

  def_named_variable("adaptive_grid_max_levels",T_INT);
  def_named_variable("adaptive_grid_max_patches",T_INT);
  def_named_variable("adaptive_grid_refinement_ratio",T_INT);
  def_named_variable("adaptive_grid_blocking_factor",T_INT);
  def_named_variable("adaptive_grid_error_buffer_cells",T_INT);
  def_named_variable("adaptive_grid_transition_cells",T_INT);
  def_named_variable("adaptive_grid_fine_residual_tolerance",T_DOUBLE);
  def_named_variable("adaptive_grid_terminal_patch_certification",T_INT);

  def_named_variable("do_photons",T_INT_2);
  def_named_variable("photon_ratio",T_DOUBLE);
  def_named_variable("ecm_min",T_DOUBLE);
  def_named_variable("ecm_min_gg",T_DOUBLE);

  def_named_variable("do_hadrons",T_INT);
  def_named_variable("store_hadrons",T_INT);
  def_named_variable("hadron_ratio",T_DOUBLE);

  def_named_variable("do_jets",T_INT);
  def_named_variable("jet_ptmin",T_DOUBLE);
  def_named_variable("jet_ratio",T_DOUBLE);
  def_named_variable("store_jets",T_INT);
  def_named_variable("jet_log",T_INT);

  def_named_variable("do_pairs",T_INT);
  def_named_variable("do_muons",T_INT);
  def_named_variable("do_coherent",T_INT);
  def_named_variable("do_trident",T_INT);
  def_named_variable("emin",T_DOUBLE);
  //def_named_variable("track_secondaries",T_INT);
  def_named_variable("track_pairs",T_INT);
  def_named_variable("track_muons",T_INT);
  def_named_variable("bhabha_ecmload",T_DOUBLE);
  //def_named_variable("store_secondaries",T_INT);
  def_named_variable("store_pairs",T_INT);
  def_named_variable("store_muons",T_INT);
  def_named_variable("pair_ecut",T_DOUBLE);
  def_named_variable("muon_ecut",T_DOUBLE);
  def_named_variable("pair_step",T_DOUBLE);
  def_named_variable("beam_size",T_INT);
  def_named_variable("beam_size_scale",T_DOUBLE);
  def_named_variable("photon_timing",T_INT);
  def_named_variable("photon_angle",T_INT);
  def_named_variable("ext_field",T_INT);
  def_named_variable("pair_ratio",T_DOUBLE);
  def_named_variable("muon_ratio",T_DOUBLE);
  def_named_variable("muon_scale",T_DOUBLE);
  def_named_variable("bhabha_ratio",T_DOUBLE);
  def_named_variable("pair_q2",T_INT);
  def_named_variable("pair_model",T_INT);
  def_named_variable("pair_exact_rr",T_DOUBLE);
  def_named_variable("photon_polarisation",T_INT);
  def_named_variable("coherent_model",T_INT);
  def_named_variable("grids",T_INT);
  def_named_variable("beam_pair",T_INT);
  def_named_variable("do_tertphot",T_INT);

  def_named_variable("electron_ratio",T_DOUBLE);
  def_named_variable("do_eloss",T_INT);
  def_named_variable("do_espread",T_INT);
  def_named_variable("do_isr",T_INT);
  def_named_variable("do_lumi",T_INT);

  def_named_variable("do_bhabhas",T_INT);
  def_named_variable("bhabha_scal",T_DOUBLE);

  def_named_variable("do_prod",T_INT);
  def_named_variable("prod_e",T_DOUBLE);
  def_named_variable("prod_scal",T_DOUBLE);

  def_named_variable("load_events",T_INT);
  def_named_variable("imported_spin_mode",T_INT);

  // def_named_variable("do_lumi_ee_2",T_INT);
  def_named_variable("hist_ee_bins",T_INT);
  def_named_variable("hist_ee_min",T_DOUBLE);
  def_named_variable("hist_ee_max",T_DOUBLE);
  def_named_variable("hist_espec_bins",T_INT);
  def_named_variable("hist_espec_min",T_DOUBLE);
  def_named_variable("hist_espec_max",T_DOUBLE);
  // def_named_variable("lumi_ee_2_n",T_INT);
  // def_named_variable("lumi_ee_2_min",T_DOUBLE);
  // def_named_variable("lumi_ee_2_max",T_DOUBLE);
  // def_named_variable("do_lumi_eg_2",T_INT);
  // def_named_variable("lumi_eg_2_n",T_INT);
  // def_named_variable("lumi_eg_2_min",T_DOUBLE);
  // def_named_variable("lumi_eg_2_max",T_DOUBLE);
  // def_named_variable("do_lumi_ge_2",T_INT);
  // def_named_variable("lumi_ge_2_n",T_INT);
  // def_named_variable("lumi_ge_2_min",T_DOUBLE);
  // def_named_variable("lumi_ge_2_max",T_DOUBLE);
  // def_named_variable("do_lumi_gg_2",T_INT);
  // def_named_variable("lumi_gg_2_n",T_INT);
  // def_named_variable("lumi_gg_2_min",T_DOUBLE);
  // def_named_variable("lumi_gg_2_max",T_DOUBLE);

  // def_named_variable("beam_vx_min",T_DOUBLE);
  // def_named_variable("beam_vx_max",T_DOUBLE);
  // def_named_variable("beam_vx_interval",T_INT);
  // def_named_variable("beam_vy_min",T_DOUBLE);
  // def_named_variable("beam_vy_max",T_DOUBLE);
  // def_named_variable("beam_vy_interval",T_INT);

  def_named_variable("do_cross",T_INT);
  def_named_variable("do_compt",T_INT);
  def_named_variable("do_compt_phot",T_INT);
  def_named_variable("compt_scale",T_DOUBLE);
  def_named_variable("compt_x_min",T_DOUBLE);
  def_named_variable("compt_emax",T_DOUBLE);
  def_named_variable("num_lumi",T_INT);
  def_named_variable("num_lumi_eg",T_INT);
  def_named_variable("num_lumi_gg",T_INT);
  def_named_variable("lumi_p",T_DOUBLE);
  def_named_variable("lumi_p_eg",T_DOUBLE);
  def_named_variable("lumi_p_gg",T_DOUBLE);
  def_named_variable("store_beam",T_INT);
  def_named_variable("load_beam",T_INT);
  def_named_variable("cuts_from_loaded_beam",T_INT);

  def_named_variable("automatic_grid_sizing",T_INT);
  def_named_variable("optimize_grids",T_INT);
  def_named_variable("do_photon_lumi",T_INT);
  def_named_variable("electron_lumi_precompute",T_INT);
  def_named_variable("electron_lumi_sampling",T_INT);
  def_named_variable("electron_lumi_spatial_kernel",T_INT);
  def_named_variable("electron_lumi_spatial_nx",T_INT);
  def_named_variable("electron_lumi_spatial_ny",T_INT);
  def_named_variable("electron_lumi_spatial_x_min",T_DOUBLE);
  def_named_variable("electron_lumi_spatial_x_max",T_DOUBLE);
  def_named_variable("electron_lumi_spatial_y_min",T_DOUBLE);
  def_named_variable("electron_lumi_spatial_y_max",T_DOUBLE);
  def_named_variable("electron_lumi_spatial_max_pairs",T_DOUBLE);
  def_named_variable("electron_lumi_spatial_closure_tolerance",T_DOUBLE);
  def_named_variable("photon_lumi_factorized",T_INT);
  def_named_variable("store_native",T_INT);
  def_named_variable("loaded_beam_is_pretransformed",T_INT);
  def_named_variable("crab_mode",T_INT_2);
  def_named_variable("crab_plane",T_INT_2);
  def_named_variable("crab_longitudinal_sign",T_INT_2);
  def_named_variable("waist_mode",T_INT_2);
  def_named_variable("crab_frequency",T_DOUBLE_2);
  def_named_variable("crab_phase",T_DOUBLE_2);
  def_named_variable("crab_target_tilt",T_DOUBLE_2);
  def_named_variable("waist_fraction",T_DOUBLE_2);
  def_named_variable("ip_dispersion",T_DOUBLE_2);
  def_named_variable("incoming_map_reslice",T_INT);
  def_named_variable("design_axis_spectrum",T_INT);
  def_named_variable("design_axis_peak_bins",T_INT);
  def_named_variable("spectrum_w_min",T_DOUBLE);
  def_named_variable("spectrum_w_max",T_DOUBLE);
  def_named_variable("spectrum_w_bins",T_INT);
  def_named_variable("crab_harmonic_n",T_INT);
  def_named_variable("crab_include_panofsky_wenzel",T_INT);
  def_named_variable("crab_full_crossing_angle",T_DOUBLE);
  def_named_variable("waist_symplecticity_tolerance",T_DOUBLE);

  def_named_variable("bmt_precession",T_INT);
  def_named_variable("ST_spin_flip",T_INT);


  def_named_variable("store_photons",T_INT);
  def_named_variable("load_photons",T_INT);
  def_named_variable("jet_pythia",T_INT);

  def_named_variable("do_dump",T_INT);
  def_named_variable("dump_step",T_INT);
  def_named_variable("dump_particle",T_INT);

 def_named_variable("do_edm4hep",T_INT);

#ifdef TWOBEAM
  def_named_variable("n2_m",T_INT_2);
#endif

  /*  define_values("{silent=1;}", m_account); */
  // Seed-only execution is independent of files left by an earlier crossing.
  // Checkpoint loading and saving are explicit opt-in operations.
  define_values("{integration_method=2;electron_distribution_rho=2;rndm_save=0;rndm_load=0;rndm_seed=1;}", m_account);
  define_values("{compute_backend=\"serial\";collision_schedule=\"legacy\";wavefront_parallel_physics=0;sparse_cells=0;wavefront_field_solver=\"fft\";sparse_direct_interaction_limit=262144;wavefront_workspace_max_bytes=0.0;cpu_threads=0;cpu_variant=\"auto\";cpu_primary_move=0;gpu_variant=\"auto\";gpu_device=0;gpu_min_particles=262144;gpu_mode=1;gpu2_scope=\"auto\";gpu2_deposition=\"auto\";gpu2_fft=\"cufft\";gpu2_streams=0;gpu2_graphs=\"auto\";gpu2_fusion=\"auto\";gpu2_batch_slices=0;gpu2_pinned_io=\"auto\";gpu2_host_threads=0;gpu2_primary_mover=\"serial\";gpu2_memory_fraction=0.80;gpu2_profile=\"summary\";rng_backend=\"legacy\";reduction_mode=\"strict\";allow_backend_fallback=0;}", m_account);
  define_values("{engine=\"legacy\";algorithm=\"exact\";precision=\"fp64\";modern_reduction=\"deterministic\";modern_batch_width=0;modern_cpu_workers=0;modern_cpu_runtime=\"auto\";modern_cpu_affinity=\"auto\";modern_gpu_streams=0;modern_gpu_graphs=\"auto\";modern_trace=\"off\";}", m_account);
  define_values("{adaptive_grid_mode=0;adaptive_grid_scope=\"encounter\";adaptive_grid_size_classes_x=\"\";adaptive_grid_size_classes_y=\"\";adaptive_grid_min_nx=0;adaptive_grid_min_ny=0;adaptive_grid_max_nx=0;adaptive_grid_max_ny=0;adaptive_grid_extra_guard_cells=0;adaptive_grid_allow_full_window=1;}", m_account);
  define_values("{adaptive_grid_max_cells=0.0;adaptive_grid_max_bytes=0.0;adaptive_grid_schedule_save=\"\";adaptive_grid_schedule_load=\"\";adaptive_grid_profile=\"summary\";adaptive_grid_pilot_diagnostics=0;}", m_account);
  define_values("{adaptive_grid_field_spacing_classes=\"\";adaptive_grid_field_operator=\"legacy_cic12_v1\";adaptive_grid_max_refinement_factor=0.0;adaptive_grid_anchor_x=0.0;adaptive_grid_anchor_y=0.0;adaptive_grid_monitor=\"charge_curvature_field_richardson_overlap\";adaptive_grid_error_target=0.0;adaptive_grid_error_norm_policy=\"component_rms_linf_v1\";adaptive_grid_error_budget_sha256=\"\";adaptive_grid_error_quantile=0.999;adaptive_grid_error_catastrophe_limit=0.5;adaptive_grid_error_reference_field=0.0;adaptive_grid_error_absolute_floor_fraction=0.0;adaptive_grid_error_catastrophe_absolute_fraction=0.0;adaptive_grid_error_exposure_limit=0.0;adaptive_grid_noise_aware_stop=0;}", m_account);
  define_values("{adaptive_grid_source_model=\"\";adaptive_grid_source_reference_nx=0.0;adaptive_grid_source_reference_ny=0.0;adaptive_grid_source_width_x=0.0;adaptive_grid_source_width_y=0.0;adaptive_grid_source_error_target=0.0;adaptive_grid_solver_error_target=0.0;adaptive_grid_lumi_error_target=0.0;adaptive_grid_lumi_floor=0.0;}", m_account);
  define_values("{adaptive_grid_tau_curvature=0.0;adaptive_grid_tau_field_gradient=0.0;adaptive_grid_tau_richardson=0.0;adaptive_grid_tau_tail=0.0;adaptive_grid_encounter_tail_fraction=0.0;adaptive_grid_rho_floor=0.0;adaptive_grid_field_floor=0.0;adaptive_grid_indicator_smoothing_passes=0;adaptive_grid_indicator_reduction=\"connected_max\";}", m_account);
  define_values("{adaptive_grid_refine_threshold=1.0;adaptive_grid_coarsen_threshold=0.35;adaptive_grid_shrink_dwell=2;adaptive_grid_update_interval=1;adaptive_grid_richardson_interval=0;adaptive_grid_richardson_order_source=\"calibrated_table\";adaptive_grid_richardson_order_table=\"\";adaptive_grid_min_neff=0.0;adaptive_grid_fail_on_unmet_error=1;}", m_account);
  define_values("{adaptive_grid_max_levels=1;adaptive_grid_max_patches=1;adaptive_grid_refinement_ratio=2;adaptive_grid_blocking_factor=0;adaptive_grid_error_buffer_cells=0;adaptive_grid_transition_cells=4;adaptive_grid_fine_residual_tolerance=0.0;adaptive_grid_terminal_patch_certification=0;}", m_account);
  define_values("{adaptive_macroparticle_mode=0;adaptive_macroparticle_domain=\"field_source\";adaptive_macroparticle_source_evolution=\"projected_current_state\";adaptive_macroparticle_decision_scope=\"encounter\";adaptive_macroparticle_commit_scope=\"wave\";}", m_account);
  define_values("{adaptive_macroparticle_strategy=\"coupled_error_budget_v1\";adaptive_macroparticle_split=\"assignment_preserving_v1\";adaptive_macroparticle_merge=\"exact_cell_projection_v1\";adaptive_macroparticle_root_source=\"physical_primary_roots_v1\";adaptive_macroparticle_root_seed=0.0;adaptive_macroparticle_scramble_id=0.0;}", m_account);
  define_values("{adaptive_macroparticle_error_target=0.0;adaptive_macroparticle_resample_bias_target=0.0;adaptive_macroparticle_min_root_support=0.0;adaptive_macroparticle_min_clone_support=0.0;adaptive_macroparticle_statistical_cohorts=1;}", m_account);
  define_values("{adaptive_macroparticle_refine_threshold=1.0;adaptive_macroparticle_coarsen_threshold=0.35;adaptive_macroparticle_update_interval=1;adaptive_macroparticle_cooldown=0;adaptive_macroparticle_dwell=2;adaptive_macroparticle_max_change_factor=4.0;adaptive_macroparticle_max_generation=32;}", m_account);
  define_values("{adaptive_macroparticle_max_count=0.0;adaptive_macroparticle_max_bytes=0.0;adaptive_macroparticle_max_weight_ratio=1000000.0;adaptive_macroparticle_policy_save=\"\";adaptive_macroparticle_policy_load=\"\";adaptive_macroparticle_trace_save=\"\";adaptive_macroparticle_trace_load=\"\";adaptive_macroparticle_profile=\"summary\";adaptive_macroparticle_fail_on_unmet_error=1;adaptive_macroparticle_determinism=\"strict\";}", m_account);
  define_values("{do_eloss=1;do_isr=0;do_espread=0;do_cross=0;}", m_account);
  define_values("{do_prod=0;prod_e=0.0;prod_scal=1e-29;load_events=0;imported_spin_mode=0;}", m_account);
  define_values("{load_beam=0;store_beam=0;load_photons=0;store_photons=0;}", m_account);
  define_values("{cuts_from_loaded_beam=0;}", m_account);

  define_values("{bmt_precession=0; ST_spin_flip=0;}", m_account);

  define_values("{automatic_grid_sizing=0;store_native=0;loaded_beam_is_pretransformed=0;optimize_grids=0;do_photon_lumi=1;electron_lumi_precompute=0;electron_lumi_sampling=0;electron_lumi_spatial_kernel=0;electron_lumi_spatial_nx=0;electron_lumi_spatial_ny=0;electron_lumi_spatial_x_min=0.0;electron_lumi_spatial_x_max=0.0;electron_lumi_spatial_y_min=0.0;electron_lumi_spatial_y_max=0.0;electron_lumi_spatial_max_pairs=0.0;electron_lumi_spatial_closure_tolerance=1e-12;photon_lumi_factorized=0;crab_mode=0;crab_plane=0;crab_longitudinal_sign=0;waist_mode=0;crab_frequency=0.0;crab_phase=0.0;crab_target_tilt=-1.0;waist_fraction=1.0;crab_harmonic_n=2;crab_include_panofsky_wenzel=1;crab_full_crossing_angle=0.0;waist_symplecticity_tolerance=1e-11;ip_dispersion=0.0;incoming_map_reslice=0;design_axis_spectrum=0;design_axis_peak_bins=10000;spectrum_w_min=0.0;spectrum_w_max=0.0;spectrum_w_bins=200;}",m_account);

  define_values("{do_photons=0;photon_ratio=1.0;electron_ratio=0.0;}", m_account);
  define_values("{do_hadrons=0;hadron_ratio=1e5;}", m_account);
  define_values("{do_jets=0;jet_ratio=1e5;store_jets=0;jet_log=1;}", m_account);
  define_values("{jet_ptmin=2.0;}", m_account);
  define_values("{beam_pair=0;}", m_account);
  //define_values("{do_pairs=0;track_secondaries=0;pair_ratio=1.0;pair_q2=2;}", m_account);
  define_values("{do_pairs=0;track_pairs=0;pair_ratio=1.0;pair_q2=2;}", m_account);
  define_values("{pair_model=0;pair_exact_rr=0.25;photon_polarisation=0;coherent_model=0;}", m_account);
  define_values("{pair_ecut=5e-3;pair_step=1.0;do_muons=0;track_muons=0;store_muons=0;}", m_account);
  define_values("{do_tertphot=0;muon_ecut=0.105;muon_ratio=1.0;muon_scale=1.0;}", m_account);
  //define_values("{beam_size=1;beam_size_scale=1.0;ext_field=0;store_secondaries=1;}", m_account);
  define_values("{beam_size=1;beam_size_scale=1.0;ext_field=0;store_pairs=0;}", m_account);
  define_values("{photon_timing=0;photon_angle=0;}", m_account);
  define_values("{do_compt=0;do_compt_phot=0;compt_x_min=1.0;}", m_account);
  define_values("{compt_emax=100.0;do_coherent=0;compt_scale=1.0;}", m_account);
  define_values("{hist_ee_bins=200;hist_ee_min=0.0;hist_ee_max=-1.0;}", m_account);
  define_values("{hist_espec_bins=-1;hist_espec_min=0.0;}", m_account);
  define_values("{hist_espec_max=500.001;}", m_account);
  //  define_values("{do_lumi_ee_2=0;}", m_account);
  define_values("{do_size_log=0;}", m_account);
  // define_values("{lumi_ee_2_n=100;lumi_ee_2_min=0.0;lumi_ee_2_max=1.0;}", m_account);
  // define_values("{do_lumi_eg_2=0;}", m_account);
  // define_values("{lumi_eg_2_n=100;lumi_eg_2_min=0.0;lumi_eg_2_max=1.0;}", m_account);
  // define_values("{do_lumi_ge_2=0;}", m_account);
  // define_values("{lumi_ge_2_n=100;lumi_ge_2_min=0.0;lumi_ge_2_max=1.0;}", m_account);
  // define_values("{do_lumi_gg_2=0;}", m_account);
  define_values("{do_lumi=0;num_lumi=100000;lumi_p=1e-29;}", m_account);
  define_values("{lumi_p_eg=1e-29;lumi_p_gg=1e-29;}", m_account);
  // define_values("{lumi_gg_2_n=100;lumi_gg_2_min=0.0;lumi_gg_2_max=1.0;}", m_account);
  define_values("{do_bhabhas=0; bhabha_ratio=1.0;bhabha_scal=1.e-29; bhabha_ecmload = 500.; ecm_min=0.0;ecm_min_gg=0.0;}", m_account);
  define_values("{cut_x=-1.0;cut_y=-1.0;cut_z=-1.0;}", m_account);
  define_values("{cut_x_factor=-1.0;cut_y_factor=-1.0;cut_z_factor=-1.0;}", m_account);
  define_values("{jet_pythia=0;}", m_account);
  // define_values("{beam_vx_min=-1e-3;beam_vx_max=1e-3;beam_vx_interval=200;}", m_account);
  // define_values("{beam_vy_min=-1e-3;beam_vy_max=1e-3;beam_vy_interval=200;}", m_account);
  define_values("{do_dump=0;dump_step=1;dump_particle=1;}", m_account);
  define_values("{do_edm4hep=0;}", m_account);
}

void input_values(char *buffer, MEMORY_ACCOUNT* m_account)
{
  TOKEN token;
  VALUE value;
  store_buffer(buffer);
  if ((token=read_token(&value))!=START){
    printf("error: no proper start\n");
    exit(1);
  }
  while((token=read_command(m_account))!=END_INPUT) {
    /*	printf("{");
	print_token(token);
	printf("}");*/
    if (token==ERROR) {
      printf("error while scanning input list\n");
      exit(1);
    }
  }
}

int file_open(DATEI *datei,char *name,char *type)
{
  if (name) {
    if(!(datei->file=fopen(name,type))) return 0;
    datei->buffer[0]='\0';
    datei->point=datei->buffer;
    datei->line_end=datei->buffer;
    return 1;
  }
  else {
    datei->file=stdin;
    datei->buffer[0]='\0';
    datei->point=datei->buffer;
    datei->line_end=datei->buffer;
    return 1;
  }
}

void file_close(DATEI *datei)
{
  if (datei->file==stdin) {
  }
  else {
    fclose(datei->file);
  }
}

void file_rewind(DATEI *datei)
{
  rewind(datei->file);
}

int file_read_line(DATEI *datei)
{
  if(!(datei->point=fgets(datei->buffer,max_buffer,datei->file))) return 0;
  datei->line_end=datei->point+strlen(datei->buffer);
  // Accept LF and CRLF without discarding the last byte of an unterminated line.
  while (datei->line_end > datei->point &&
         (datei->line_end[-1] == '\n' || datei->line_end[-1] == '\r')) {
    *(--datei->line_end) = '\0';
  }
  return 1;
}

int file_find_word_line(DATEI *datei,char *word)
{
  if(!(datei->point=strstr(datei->point,word))) return 0;
  datei->point+=strlen(word);
  return 1;
}

int file_find_word(DATEI *datei,char *word)
{
  if (file_find_word_line(datei,word)) return 1;
  while(file_read_line(datei))
    {
      if(file_find_word_line(datei,word)) return 1;
    }
  return 0;
}

int file_skip_space_line(DATEI *datei)
{
  if(datei->point==NULL) return 0;
  (datei->point)--;
  while((++(datei->point)) < datei->line_end)
    if (!std::isspace(static_cast<unsigned char>(*(datei->point)))) return 1;
  return 0;
}

int file_skip_space(DATEI *datei)
{
  if (file_skip_space_line(datei)) return 1;
  while(file_read_line(datei))
    if (file_skip_space_line(datei)) return 1;
  return 0;
}

int file_next_word(DATEI *datei,char *word)
{
  if(!file_skip_space(datei)) return 0;
  if (strncmp(datei->point,word,strlen(word))!=0) return 0;
  datei->point+=strlen(word);
  return 1;
}

double file_get_double(DATEI *datei)
{
  if(!file_skip_space(datei)) return 0;
  return strtod(datei->point,&(datei->point));
}

int file_read_double(DATEI *datei,double *x)
{
  if(!file_skip_space(datei)) return 0;
  *x=strtod(datei->point,&(datei->point));
  return 1;
}

int file_read_int(DATEI *datei,int *n)
{
  if(!file_skip_space(datei)) return 0;
  *n=strtol(datei->point,&(datei->point),10);
  return 1;
}

int file_read_until(DATEI *datei,char *end,char *buff,int n_max)
{
  char *tmp;
  int len;
  buff[0]='\0';
  len=strlen(end);
  while((tmp=strstr(datei->point,end))==NULL)
    {
      if((n_max-=strlen(datei->point))>1)
	{
	  buff=stpcpy(buff,datei->point);
	  //buff=strcpy(buff,datei->point);
	}
      else
	{
	  return 0;
	}
      if (!file_read_line(datei))
        throw std::invalid_argument("Input block ended before its closing brace");
    }
  if((n_max-=(tmp-datei->point)+len)>1)
    {
      strncat(buff,datei->point,tmp-datei->point+len);
    }
  else{
  }
  datei->point=tmp+len;
  return 1;
}

int file_read_braces(DATEI *datei,char *begin,char *end,char *buff,int n_max)
{
  if(!file_skip_space(datei)) return 0;
  if(strncmp(datei->point,begin,strlen(begin))!=0) return 0;
  return file_read_until(datei,end,buff,n_max);
}

int file_find_braces(DATEI *datei,char *begin,char *end,char *buff,int n_max)
{
  if (!file_find_word(datei,begin)) return 0;
  if (!file_read_until(datei,end,buff,n_max)) return 0;
  return 1;
}

int string_get_int(char *line,char *word,int *n)
{
  char *pointer;

  if ((pointer=strstr(line,word))==NULL) return 0;
  pointer+=strlen(word);
  while (*pointer==' ') pointer++;
  if (*pointer!='=') return 0;
  pointer++;
  while (*pointer==' ') pointer++;
  *n=strtol(pointer,&pointer,10);
  return 1;
}

int string_get_float(char *line,char *word,float *x)
{
  char *pointer;

  if ((pointer=strstr(line,word))==NULL) return 0;
  pointer+=strlen(word);
  while (*pointer==' ') pointer++;
  if (*pointer!='=') return 0;
  pointer++;
  while (*pointer==' ') pointer++;
  *x=strtod(pointer,&pointer);
  return 1;
}

int string_get_list(char *line,char *word,float *x,int *n)
{
  char *pointer;
  int i=0;

  if ((pointer=strstr(line,word))==NULL) return 0;
  pointer+=strlen(word);
  while (*pointer==' ') pointer++;
  if (*pointer!='=') return 0;
  pointer++;
  while (*pointer==' ') pointer++;
  if(*pointer!='(') return 0;
  pointer++;
  while (*pointer!=')'){
     x[i]=strtod(pointer,&pointer);
     i++;
     pointer++;
  }
  *n=i;
  return 1;
}

void imprimerBuffer(char* buffer)
{
 int k=0;
  while (buffer[k] != '\0') 
    {
      printf(" %c ", buffer[k]);
      k++;
    }
  printf(" \n");
}
