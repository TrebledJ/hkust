//
//  assignment2_assertions.hpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#ifndef PA2_ASSERTIONS_H
#define PA2_ASSERTIONS_H

//
//  include in main assignment and call Assertions::test()
//  to test testcases for testing
//


unsigned int recursive_strlen(const char line[], int start);
unsigned int count_dquotes(const char line[], int start);
int find_first_dquote(const char line[], int start);
int count_chars_in_matched_dquote(const char line[], int start);
bool check_quotes_matched(const char line[], int start);
unsigned int length_of_longest_consecutive_dquotes(const char line[], int start);


namespace Assertions {
    void test() {
        assert(recursive_strlen("\0", 0) == 0);
        assert(recursive_strlen("A\0", 0) == 1);
        assert(recursive_strlen("COMP2011\0", 0) == 8);
        
        assert(count_dquotes("\0", 0) == 0);
        assert(count_dquotes("COMP2011\0", 0) == 0);
        assert(count_dquotes("\"COMP2011\"\0", 0) == 2);
        
        assert(find_first_dquote("\0", 0) == ERROR);
        assert(find_first_dquote("COMP2011\0", 0) == ERROR);
        assert(find_first_dquote("\"COMP2011\"\0", 0) == 0);
        assert(find_first_dquote("Hello \" World\0", 0) == 6);
        
        assert(count_chars_in_matched_dquote("\0", 0) == 0);
        assert(count_chars_in_matched_dquote("COMP2011\0", 0) == 0);
        assert(count_chars_in_matched_dquote("\"COMP2011\"\0", 0) == 8);
        assert(count_chars_in_matched_dquote("Hello \" World\0", 0) == ERROR);
        assert(count_chars_in_matched_dquote("\"Hello\" \"World\"\0", 0) == 10);
        assert(count_chars_in_matched_dquote("\" \"Hello World\" \"\0", 0) == 2);
        assert(count_chars_in_matched_dquote("\"A double quote: \"\"\0", 0) == ERROR);
        
        assert(check_quotes_matched("\0", 0) == true);
        assert(check_quotes_matched("COMP2011\0", 0) == true);
        assert(check_quotes_matched("'COMP2011'\0", 0) == true);
        assert(check_quotes_matched("Hello \" World\0", 0) == false);
        assert(check_quotes_matched("\"A single quote: '\"\0", 0) == true);
        assert(check_quotes_matched("\" 'Hello World' \"\0", 0) == true);
        assert(check_quotes_matched("\"A'A'A'A\"\0", 0) == true);
        assert(check_quotes_matched("'\"A\"\"A\"\0", 0) == false);
        
        assert(length_of_longest_consecutive_dquotes("\0", 0) == 0);
        assert(length_of_longest_consecutive_dquotes("COMP2011\0", 0) == 0);
        assert(length_of_longest_consecutive_dquotes("\"COMP2011\"\0", 0) == 1);
        assert(length_of_longest_consecutive_dquotes("Hello \" World\0", 0) == 1);
        assert(length_of_longest_consecutive_dquotes("\"\"\"Hello World\"\"\"\0", 0) == 3);
        assert(length_of_longest_consecutive_dquotes("AAAAAA\"\"BBB\0", 0) == 2);
        assert(length_of_longest_consecutive_dquotes("\"\"\"AAA\"\"\"\"\"\"\0", 0) == 6);
        assert(length_of_longest_consecutive_dquotes("\"\"\"\"\"BBBBBBBBB\"\"\"\0", 0) == 5);
    }
}


#endif
