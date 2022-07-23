%option noyywrap

%{
#define YYSTYPE void*
#define MAXL 5000
#include "SMILEScal.tab.h"
%}

whitespace                    [ \t]+
integer                       [0-9]+
bond                          \-|\=|\#|\$|\/|\\
digit                         [0-9]
aliphatic_organic             B|C|N|O|S|P|F|Cl|Br|I
aromatic_organic              b|c|n|o|s|p
symbol                        H|He|Li|Be|Ne|Na|Mg|Al|Si|Ar|K|Ca
hydrogen                      H
left_brace_bucket             [\{]
right_brace_bucket            [\}]
left_square_bucket            [\[]
right_square_bucket           [\]]
left_circle_bucket            [(]
right_circle_bucket           [)]
ari_cou                       [\|]
ari_add                       [\+]
hattached                     {hydrogen}|{hydrogen}{digit}
bracket_atom_substructure     {left_square_bucket}({symbol}|{aliphatic_organic})?({hattached})?{right_square_bucket}
atom                          {bracket_atom_substructure}|{aliphatic_organic}|{aromatic_organic}
branch                        {left_circle_bucket}(({chain})*)+{right_circle_bucket}
chain                         (({bond})?{atom})+
smiles                        {atom}({chain}|{branch})*
%%
{left_brace_bucket}     {return LBB;}
{right_brace_bucket}    {return RBB;}
{left_square_bucket}    {return LSB;}
{right_square_bucket}   {return RSB;}
{left_circle_bucket}    {return LCB;}
{right_circle_bucket}   {return RCB;}
{ari_cou}               {return COUNT;}
{ari_add}               {return ADD;}
"MAX"	                {return MAX;}
"MIN"	                {return MIN;}
"UNIQUE"	            {return UNIQUE;}
{smiles}                {yylval=(void*)malloc(sizeof(char)*MAXL);strcpy(yylval,yytext); return SMILES;}
{integer}               {yylval=(void*)atol(yytext); return INT;}
\n                      {return *yytext;}
{whitespace}            /* ignore white spaces */
%%