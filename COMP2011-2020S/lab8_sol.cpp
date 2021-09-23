#include <iostream>
#include <fstream>
using namespace std;

bool isSameCourse(char c1[], char c2[]){

    int i=0;
    int j=0;
    while(c1[i]!='\0' && c2[j]!='\0'){
        if(c1[i]!=c2[j]){
            return false;
        }
        i++;j++;
    }

    if(c1[i]=='\0' && c2[j]=='\0'){
        return true;
    }

    return false;

}

void addCourse(char name[], char course[]){

    cout << "Adding course: " << course <<endl;

     ifstream infile;
    infile.open(name);

    if(infile){
        char data[1000];
        while(infile >> data){
            if(isSameCourse(data,course)){
                cout << "This course has been alreay added" << endl;
                return;
            }
        }

        infile.close();
    }
    else{
        cout << "This is the first try of the student" << endl;
    }

    ofstream outfile;
    outfile.open(name, ios::app);

    outfile << course << endl;
    outfile.close();

    cout << "Add course successfully" << endl;

}

void dropCourse(char name[], char course[]){

    cout << "Dropping course: " << course <<endl;

    ifstream infile;
    infile.open(name);

    if(!infile){
        cout << "Cannot find student: " << name <<", drop failed!" << endl;
        return;
    }

    char courses[11][200];
    int pointer = 0;
    bool found = false;
    while(infile >> courses[pointer]){
        if(courses[pointer]==NULL || courses[pointer][0]=='\0')break;
        
        if(!isSameCourse(courses[pointer],course)){
            pointer++;
        }else{
            found = true;

        }

    }

    if(!found){
        cout << "The student has not enrolled the course, cannot drop." <<endl;
        return;
    }


    infile.close();

    ofstream outfile;
    outfile.open(name);
    
    for(int i=0;i<pointer;++i){
        outfile << courses[i] << endl;
    }

    outfile.close();
    cout << "Drop course successfully" << endl;

    

}

void listCourse(char name[]) {

    cout << "Listing course of the student: " << name <<endl;

    ifstream infile;
    infile.open(name);

    if(!infile){
        cout << "Cannot find student: " << name <<", list failed!" << endl;
        return;
    }

    char data[20];
    bool hasCourse = false;
    while(infile >> data){
        hasCourse = true;
        cout << data << ' ';
    }

    if(hasCourse){
        cout << endl;
    }else{
        cout << "no course enrolled." << endl;
    }
    

}

int main() {

    char action;
    char name[20];
    char course[20];

    while(true){
        cout << "----------------------" <<endl;
        cout << "Add-Drop course start!" << endl;
        cout << "A for add; D for drop; L for list; Q for quit:" << endl;

        cin >> action;
        cin.get();

        if(action == 'A'){
            cout << "Please enter student name: " << endl;
            cin.getline(name, sizeof(name));
            if(name[0] == '\0'){
                cout << "name cannot be empty!" << endl;
                continue;
            }
            cout << "Please enter the course you want to add: " << endl;
            cin.getline(course, sizeof(course));
            if(course[0] == '\0'){
                cout << "course cannot be empty!" << endl;
                continue;
            }

            addCourse(name,course);
        }
        else if(action == 'D'){
            cout << "Please enter student name: " << endl;
            cin.getline(name, sizeof(name));
            if(name[0] == '\0'){
                cout << "name cannot be empty!" << endl;
                continue;
            }
            cout << "Please enter the course you want to drop: " << endl;
            cin.getline(course, sizeof(course));
            if(course[0] == '\0'){
                cout << "course cannot be empty!" << endl;
                continue;
            }

            dropCourse(name,course);

        }else if(action == 'L'){
            cout << "Please enter student name: " << endl;
            cin.getline(name, sizeof(name));
            if(name[0] == '\0'){
                cout << "name cannot be empty!" << endl;
                continue;
            }

            listCourse(name);
        }else if(action == 'Q'){
            cout << "Quit..." << endl;
            break;
        }
        else{
            cout << "Input Invalid, please input again" << endl;
        }

    }

    return 0;

}
