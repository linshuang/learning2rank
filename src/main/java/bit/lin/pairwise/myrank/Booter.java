package bit.lin.pairwise.myrank;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.RandomAccessFile;

public class Booter {
	private static String _valiSet = "/home/lins/data/learning to rank/10k4j/Fold1/vali.txt";

	public static void train(int pos) {
		try {
			PrfIntervalClassifier pic = restorePIC();
			Trainer t = new Trainer(pic);
			t.run(10, 1, pos);
			int count = 1;
			while (pic.isTraing()) {
				Thread.sleep(count * 10000);
				if (count != 20)
					count++;
			}
			FileOutputStream fos = new FileOutputStream(
					"src/main/resources/pic");
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(pic);
			oos.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found while storing Neural Net.");
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("IO error while storing Neural Net.");
			e.printStackTrace();
		} catch (InterruptedException e) {
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void vali() {
		try {
			PrfIntervalClassifier pic =restorePIC();
			RandomAccessFile raf = new RandomAccessFile(_valiSet, "r");
			for (int i = 1000; i > 0; i--) {
				String line1 = raf.readLine();
				String line2 = raf.readLine();
				int rslt = pic.getPrf(line1, line2);
				System.out.format("score %d - score %d - result %d\n",
						new Integer(line1.split(";")[0]),
						new Integer(line2.split(";")[0]), rslt);
			}
		} catch (FileNotFoundException e) {
			System.out.println("File not found while restoring Neural Net.");
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("IO error while restoring Neural Net.");
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static PrfIntervalClassifier restorePIC() throws IOException,
			ClassNotFoundException {
		FileInputStream fis = new FileInputStream("src/main/resources/pic");
		ObjectInputStream ois = new ObjectInputStream(fis);
		return (PrfIntervalClassifier) ois.readObject();
	}

	public static void main(String[] args) {
		Booter.train(50);
//		Booter.vali();
	}
}
