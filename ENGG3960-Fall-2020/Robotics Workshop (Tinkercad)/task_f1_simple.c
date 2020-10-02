// 
//  task_f1_simple.c
//  https://www.tinkercad.com/things/lBKUoZ2XJcx-workshop-f1-task
// 

#include <time.h>

//  Define a set of states which allow us to 
//  conveniently write stuff like `state = BLANK;`.
enum State {
  COUNTDOWN,
  FORMATION,
  BLANK,
  GO,

  NUM_STATES,
} prev_state, state = FORMATION; //  The current state, initially set.
//  Try changing the initial state from `state = COUNTDOWN` to `state = FORMATION;`
//  To see a different sequence of lights.

//  Define some constants.
const int NUM_LEDS = 5;
const int RED_LED[] = {4, 3, 2, 1, 0}; // Red LED Pins, ordered from LEFT to RIGHT.
const int GREEN_LED[] = {10, 9, 8, 7, 6}; //  Green LED Pins, ordered from LEFT to RIGHT.

const int COUNTDOWN_DELAY = 1000;
const int FORMATION_DELAY = 2000;

bool state_changed();
bool led_update_needed();

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

  prev_state = state;
}

void loop()
{
  //  Use a static variable to keep track of counting.
  static int counter = 0;

  if (counter == NUM_LEDS)
  {
    counter = 0;
    state = (state == COUNTDOWN ? BLANK 
           : state == FORMATION ? GO
           : BLANK);
    delay(COUNTDOWN_DELAY + rand() % 4000); //  Delay an additional 0-4 seconds.
  }

  if (led_update_needed())
  {
    if (state_changed())
    {
      //  Reset the counter.
      counter = 0;
    }

    switch (state)
    {
    case COUNTDOWN:
      for (int i = 0; i < NUM_LEDS; ++i)
      {
        digitalWrite(RED_LED[i], (i <= counter));
        digitalWrite(GREEN_LED[i], LOW);
      }
      counter++;
      delay(COUNTDOWN_DELAY);
      break;

    case FORMATION:
      for (int i = 0; i < NUM_LEDS; ++i)
      {
        digitalWrite(RED_LED[i], (i >= counter));
        digitalWrite(GREEN_LED[i], LOW);
      }
      counter++;
      delay(FORMATION_DELAY);
      break;

    case BLANK:
      for (int i = 0; i < NUM_LEDS; ++i)
      {
        digitalWrite(RED_LED[i], LOW);
        digitalWrite(GREEN_LED[i], LOW);
      }

      break;

    case GO:
      for (int i = 0; i < NUM_LEDS; ++i)
      {
        digitalWrite(RED_LED[i], LOW);
        digitalWrite(GREEN_LED[i], HIGH);
      }
      break;

    default:
      break;
    }
  }

  prev_state = state;
}

bool state_changed()
{
  return prev_state != state;
}

bool led_update_needed()
{
  return prev_state != state || state == COUNTDOWN || state == FORMATION;
}