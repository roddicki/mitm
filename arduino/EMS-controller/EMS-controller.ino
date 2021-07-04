// this script controls 12 relays connected to an Arduino that turn on and off the EMS pads placed on the subject's face
// see video for rough guide to placement https://vimeo.com/569320366
// the script StreamMetrics-ouput-1.py communicates with the Arduino as it runs the ML emotion detection

int x;
const uint8_t _OFF = HIGH;
const uint8_t _ON = LOW;


// the setup function runs once when you press reset or power the board
void setup() {
  // serial set up
  Serial.begin(115200);
  Serial.setTimeout(1);
  // initialize digital pin _BUILTIN as an output.
  pinMode(2, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(12, OUTPUT);
  //pinMode(2, INPUT_PULLUP);
  turnOff();
}

// the loop function runs over and over again forever
void loop() {
  while (!Serial.available());
  x = Serial.readString().toInt();

  // Off
  if(x == 0) {
    turnOff();
    Serial.print("Turn off TENS");
  }

  // Happy
  if(x == 1) {
    turnOff();
    Serial.print("Turn on TENS");
    delay(1000);
    digitalWrite(2, _ON);   // turn the  on (HIGH is the voltage level)
    digitalWrite(4, _ON);   // turn the  on (HIGH is the voltage level)
  }

  // Sad
  if(x == 2) {
    turnOff();
    Serial.print("Turn on TENS");
    delay(1000);
    digitalWrite(7, _ON);   // turn the  on (HIGH is the voltage level)
    digitalWrite(8, _ON);   // turn the  on (HIGH is the voltage level)
  }

  // Stressed
  if(x == 3) {
    turnOff();
    Serial.print("Turn on TENS");
    delay(1000);
    digitalWrite(12, _OFF);   // turn the  on (HIGH is the voltage level)
  }

}

void turnOff() {
  digitalWrite(2, _OFF);
  digitalWrite(4, _OFF);
  digitalWrite(7, _OFF);
  digitalWrite(8, _OFF);
  digitalWrite(12, _ON);
}
  
 
