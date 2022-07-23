%{
#define YYSTYPE void*
#include <stdio.h>
#include "helpers.h"
%}

/********** Start: add your tokens here **********/
%token LBB RBB LSB RSB LCB RCB COUNT ADD MAX MIN UNIQUE SMILES INT
 /********** End: add your tokens here **********/

%%

/********** Start: add your grammar rules here **********/
input:  /* empty */
    |   input line
	;

line:   '\n'
    |   expr '\n' {printElem($1);};
	

expr: expr_sm {$$=$1;}
	| expr_sm COUNT expr_sm {$$=count_atom($1,$3);}
	| UNIQUE expr_sm {$$=unique_atom($2);}
	| MAX expr_sm {$$=max_atom($2);}
	| MIN expr_sm {$$=min_atom($2);}
	| expr ADD expr {$$=addNum($1,$3);}
	| LBB expr RBB {$$=$2;}
	;


/********** End: add your grammar rules here **********/

expr_sm:  SMILES {$$=generateSmiles($1);};
%%

int main() { return yyparse(); }
int yyerror(const char* s) { printf("%s\n", s); return 0; }
