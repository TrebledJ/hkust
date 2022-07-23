%{
#define YYSTYPE void*
#include <stdio.h>
#include "helpers.h"

int yyerror(const char *s);
int yylex(void);
%}

/********** Start: add your tokens here **********/

// Token kinds.
%token SMILES

// Operator precedence and associativity.
%left '|'
%right UNIQUE
%right MIN MAX
%left '+'

/********** End: add your tokens here **********/

%%

/********** Start: add your grammar rules here **********/

// Input handling.
input:          /* empty */ | input line
line:           '\n' 
    |           expr '\n' {printElem($1);}
    ;


// SMILES operations. Direct correspondence with expression grammar.
expr:   expr_sm
    |   expr_sm '|' expr_sm     {$$ = count_atom($1, $3);}
    |   UNIQUE expr_sm          {$$ = unique_atom($2);}
    |   MAX expr_sm             {$$ = max_atom($2);}
    |   MIN expr_sm             {$$ = min_atom($2);}
    |   expr '+' expr           {$$ = addNum($1, $3);}
    |   '{' expr '}'            {$$ = $2;}
    ;

/********** End: add your grammar rules here **********/

expr_sm:  SMILES {$$=generateSmiles($1);};
%%

int main() { return yyparse(); }
int yyerror(const char* s) { printf("%s\n", s); return 0; }
