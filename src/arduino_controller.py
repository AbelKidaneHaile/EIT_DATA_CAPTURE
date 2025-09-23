import pyfirmata
import time

# Port number declaration 
L1_In = 2  # Relay One
L1_Ex = 3  # Relay Two
L2_In = 4  # Relay Three
L2_Ex = 5  # Relay Four

class ArduinoController:
    def __init__(self, port="COM5"):
        self.port = port
        self.board = None
        self.pin_inflate = 9  # Pin for inflation control
        self.pin_deflate = 10  # Pin for deflation control

    def connect(self):
        try:
            self.board = pyfirmata.Arduino(self.port)
            it = pyfirmata.util.Iterator(self.board)
            it.start()
            print(f"[INFO] Connected to Arduino on {self.port}")
            
            self.board.digital[L1_In].write(0)
            self.board.digital[L1_Ex].write(0)
            self.board.digital[L2_In].write(0)
            self.board.digital[L2_Ex].write(0)
        except Exception as e:
            print(f"[ERROR] Could not connect to Arduino: {e}")
            self.board = None

    def disconnect(self):
        L1_In = 2  # Relay One
        L1_Ex = 3  # Relay Two
        L2_In = 4  # Relay Three
        L2_Ex = 5  # Relay Four
        if self.board:
            self.board.digital[L1_In].write(0)
            self.board.digital[L1_Ex].write(0)
            self.board.digital[L2_In].write(0)
            self.board.digital[L2_Ex].write(0)
            time.sleep(1)  # Ensure all commands are sent before exiting            
            self.board.exit()
            print("[INFO] Disconnected from Arduino")

    def inflate(self, duration):
        L1_In = 2  # Relay One
        L1_Ex = 3  # Relay Two
        L2_In = 4  # Relay Three
        L2_Ex = 5  # Relay Four
        if not self.board:
            print("[ERROR] Not connected to Arduino")
            return
        print(f"[INFO] Inflating for {duration} seconds...")
        # self.board.digital[self.pin_inflate].write(1)  # Set pin high
        self.board.digital[L1_In].write(1)
        self.board.digital[L1_Ex].write(0)
        self.board.digital[L2_In].write(1)
        self.board.digital[L2_Ex].write(0)
        
        time.sleep(duration)
        
        # self.board.digital[self.pin_inflate].write(0)  # Set pin low
        self.board.digital[L1_In].write(0)
        self.board.digital[L1_Ex].write(0)
        self.board.digital[L2_In].write(0)
        self.board.digital[L2_Ex].write(0)
        
        print("[INFO] Inflation complete")

    def deflate(self, duration):
        if not self.board:
            print("[ERROR] Not connected to Arduino")
            return
        print(f"[INFO] Deflating for {duration} seconds...")
        # self.board.digital[self.pin_deflate].write(1)  # Set pin high
        self.board.digital[L1_In].write(0)
        self.board.digital[L1_Ex].write(1)
        self.board.digital[L2_In].write(0)
        self.board.digital[L2_Ex].write(1)
        
        time.sleep(duration)
        
        # self.board.digital[self.pin_deflate].write(0)  # Set pin low
        self.board.digital[L1_In].write(0)
        self.board.digital[L1_Ex].write(0)
        self.board.digital[L2_In].write(0)
        self.board.digital[L2_Ex].write(0)
        
        print("[INFO] Deflation complete")
