// 
//  rising_edge.c
// 
//  This file gives two examples of how to detect the 
//  rising edge of a pushbutton press.
// 

// 
// Example 1
// 

//  Global variables.
bool prev_pressed = false;  //  Stores the previous pressed state.
int state = 0;  //  Stores the state of the traffic lights (RED, RED + YELLOW, GREEN, YELLOW).

void loop()
{
  //  Update state based on button press (one-time per hold).
  bool pressed = digitalRead(13);
  if (!prev_pressed && pressed)
  {
    state = (state + 1) % 4; // advance the state
  }
  prev_pressed = pressed;

  if (state == 0)
  {
    //  Display 1st state: RED
  }
  else if (state == 1)
  {
    //  Display 2nd state: YELLOW
  }
  //  and so on...
}


// 
// Example 2
// 

void loop()
{
  if (digitalRead(13))
    state = (state + 1) % 4; // advance the state

  while (digitalRead(13)) // wait while pressed/held
  {
    delay(10); // delay a bit between checks
  }

  delay(10);  //  give a small additional delay in case the button was quickly released
}