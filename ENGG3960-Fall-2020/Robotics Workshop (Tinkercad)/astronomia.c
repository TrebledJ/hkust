// 
//  astronomia.c
//  https://www.tinkercad.com/things/lzfTeU1HISW-astronomia
// 

#include <time.h>


#define ENABLE_BUZZER     //  Comment to disable buzzer.


/**
 * Enums and Global Variables
 **/
//  Define a set of states which allow us to 
//  conveniently write stuff like `state = BLANK;`.
enum State {
  COUNTDOWN,
  FORMATION,
  BLANK,
  GO,

  NUM_STATES,
} prev_state, state = FORMATION; //  The current state, initially set.

//  Define some constants.
const int NUM_LEDS = 5;
const int RED_LED[] = {4, 3, 2, 1, 0}; // Red LED Pins, ordered from LEFT to RIGHT.
const int GREEN_LED[] = {10, 9, 8, 7, 6}; //  Green LED Pins, ordered from LEFT to RIGHT.
const int BUZZER = 11;

const int COUNTDOWN_DELAY = 1000;
const int FORMATION_DELAY = 3000;
int counter = 0;  //  Use a variable to keep track of counting.


/**
 * Music Stuff
 **/
#define C4 261.63
#define Cs4 277.18
#define D4 293.66
#define Ds4 311.13
#define E4 329.63
#define F4 349.23
#define Fs4 369.99
#define G4 392.00
#define Gs4 415.30
#define A4 440.00
#define As4 466.16
#define B4 493.88
#define C5 523.25
#define Cs5 554.37
#define D5 587.33
#define Ds5 622.25
#define E5 659.25
#define F5 698.46
#define Fs5 739.99
#define G5 783.99
#define Gs5 830.61
#define A5 880.00
#define As5 932.33
#define B5 987.77
#define C6 1046.50
#define Cs6 1108.73
#define D6 1174.66
#define Ds6 1244.51
#define E6 1318.51
#define F6 1396.91
#define Fs6 1479.98
#define G6 1567.98
#define Gs6 1661.22
#define A6 1760.00
#define As6 1864.66
#define B6 1975.53

#define Bb4 As4
#define Bb5 As5
#define Bb6 As6

typedef struct Note
{
  float frequency;
  float duration;
} Note;

#define N(p, d) (Note){p, d}

typedef struct Melody
{
  Note* notes;
  int size;
} Melody;

#define DEFINE_MELODY(name, ...) static Note name##_notes[] = {__VA_ARGS__}; Melody name = (Melody){name##_notes, sizeof(name##_notes) / sizeof(Note)};

const int bpm = 120;
DEFINE_MELODY(
    astronomia,
    N(C5, .5), N(Bb4, .5), N(A4, .5), N(F4, .5),
    N(G4, 1), N(G4, .5), N(D5, .5), N(C5, 1), N(Bb4, 1),
    N(A4, 1), N(A4, .5), N(A4, .5), N(C5, 1), N(Bb4, 0.5), N(A4, 0.5),
    N(G4, 1), N(G4, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5),
    N(G4, 1), N(G4, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5),

    N(G4, 1), N(G4, .5), N(D5, .5), N(C5, 1), N(Bb4, 1),
    N(A4, 1), N(A4, .5), N(A4, .5), N(C5, 1), N(Bb4, 0.5), N(A4, 0.5),
    N(G4, 1), N(G4, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5),
    N(G4, 1), N(G4, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5),

    N(G4, 1), N(G4, .5), N(D5, .5), N(C5, 1), N(Bb4, 1),
    N(A4, 1), N(A4, .5), N(A4, .5), N(C5, 1), N(Bb4, 0.5), N(A4, 0.5),
    N(G4, 1), N(G4, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5),
    N(G4, 1), N(G4, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5), N(A5, .5), N(Bb5, .5),

    N(Bb4, 0.5), N(Bb4, 0.5),  N(Bb4, 0.5),  N(Bb4, 0.5),  N(D5, 0.5),  N(D5, 0.5), N(D5, 0.5), N(D5, 0.5), 
    N(C5, 0.5), N(C5, 0.5),  N(C5, 0.5),  N(C5, 0.5),  N(F5, 0.5),  N(F5, 0.5), N(F5, 0.5), N(F5, 0.5), 

    N(G5, 0.5), N(G5, 0.5),  N(G5, 0.5),  N(G5, 0.5),  N(G5, 0.5),  N(G5, 0.5), N(G5, 0.5), N(G5, 0.5), 

    N(G5, 0.5), N(G5, 0.5),  N(G5, 0.5),  N(G5, 0.5),
);

inline float time_of(const struct Note& note)
{
  return 60e3 * note.duration / bpm;
}

inline void play_note(const struct Note& note)
{
  tone(BUZZER, note.frequency, time_of(note));
}

/**
 * Main Arduino Code
 **/
void setup()
{
  //  Seed randomness.
  srand(time(0));

  //  Initialise the LEDs.
  for (int i = 0; i < NUM_LEDS; ++i) 
  {
    pinMode(RED_LED[i], OUTPUT);
    pinMode(GREEN_LED[i], OUTPUT);
  }
  pinMode(BUZZER, OUTPUT);

  prev_state = state;
}

void loop()
{
  static uint32_t next_change = 0;
  static uint32_t next_led_change = 0;
  static uint32_t music_index = 0;

  uint32_t now = millis();
  if (now > next_change)
  {
    Note note = astronomia.notes[music_index];
    music_index = (music_index + 1) % astronomia.size;
    play_note(note);
    next_change = now + time_of(note);
  }
  
  if (now > next_led_change)
  {
    for (int i = 0; i < NUM_LEDS; ++i)
    {
      digitalWrite(RED_LED[i], rand() % 2);
      digitalWrite(GREEN_LED[i], (rand() * 31 + 107) % 2);
    }
    next_led_change = now + 100;
  }

  delay(20);
}