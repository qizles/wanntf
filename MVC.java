import java.awt.*;
import java.awt.event.*;

/**
 *  @author
 *  @version 0.1
 *
 */
public class MVC extends Frame implements WindowListener, ActionListener {
        //  Textfields to enter and display text
        TextField a = new TextField(30);
        TextField b = new TextField(30);

        //  Clicker class extends Button
        Button clicker;
        //  Counter for special first case
        private int clickCounter = 0;

        //  Setup Window and main function
        public static void main(String[] args) {
                MVC myWindow = new MVC("Clicker");
                myWindow.setSize(800,300);
                myWindow.setVisible(true);
        }

        /**
         *  Constructor
         *  @param  title: Label of the Button
         */
        public MVC(String title) {
                super(title);
                setLayout(new FlowLayout());
                addWindowListener(this);
                clicker = new Button("Button");
                add(a);
                b.setEnabled(false);
                add(b);
                add(clicker);
                clicker.addActionListener(this);
                a.addActionListener(this);
        }

        /**
         *  Function is called when button is pressen
         *  @param e: the action event
         */
        public void actionPerformed(ActionEvent e) {
          if (clickCounter == 0 && a.getText().isEmpty()) {
            return;
          }

          if (clickCounter == 0) {
            clickCounter++;
            b.setText("Nothing");
          } else {
            clickCounter++;
            b.setText(a.getText());
          }
        }

        /**
         *  Function is called when window is closed
         *  @param e: the action event
         */
        public void windowClosing(WindowEvent e) {
                dispose();
                System.exit(0);
        }

        public void windowOpened(WindowEvent e) {}
        public void windowActivated(WindowEvent e) {}
        public void windowIconified(WindowEvent e) {}
        public void windowDeiconified(WindowEvent e) {}
        public void windowDeactivated(WindowEvent e) {}
        public void windowClosed(WindowEvent e) {}

}
