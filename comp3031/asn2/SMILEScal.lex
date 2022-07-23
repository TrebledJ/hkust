%option noyywrap

%{
#define YYSTYPE void*
#define MAXL 5000
#include "SMILEScal.tab.h"
%}


/********** Start: add your definitions here **********/

/* Lexing labels. */
/* This is simplified with EBNF so that flex doesn't complain about complicated input rules. */
smiles                      {atom}({branch}|{chain})*
branch                      \({chain}\)
chain                       ({bond}?{atom})+
atom                        {bracket_atom_substructure}|{aliphatic_organic}|{aromatic_organic}

bracket_atom_substructure   \[({symbol}|{aliphatic_organic}){hattached}?\]|\[{hattached}\]
aliphatic_organic           B|C|N|O|S|P|F|Cl|Br|I
aromatic_organic            [bcnosp]
symbol                      H|He|Li|Be|Ne|Na|Mg|Al|Si|Ar|K|Ca
hattached                   H{digit}?

digit                       [0-9]
bond                        [-=#$/\\]

op                          [+|{}]
ws                          [ \t]+

/********** End: add your definitions here **********/

%%

{smiles}    yylval = yytext; return SMILES; /* Pass on the matched smiles string to bison. */
UNIQUE      return UNIQUE; /* Return tags that Bison will recognise as terminals. */
MAX         return MAX;
MIN         return MIN;
{op}|\n     return *yytext; /* Simply return the character. We'll parse these as char literals in Bison. */
{ws}        /* discard */

%%

/*

Test Cases
----------

CCC=O|C
[CH4]|[H]
UNIQUE N[CH](C)C(=O)O
MAX [CH4]
{CCC=O|C}+{CCC(CC)CO|C}
CCC=O
{UNIQUE CCC=O}+{MAX CCC(CC)CO}

3
4
4
4
9
4
8

*/
