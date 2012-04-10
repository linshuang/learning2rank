package bit.lin.utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import org.joone.engine.Layer;
import org.joone.io.MemoryInputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;

/**
 * Utils to use joone more easily
 */
public class JUtils {
	/**
	 * used to store org.joone.net.NeuralNet into a specific file
	 */
	public void saveNeuralNet(NeuralNet nn, String fName) {
		int count = 1;
		try {
			while (nn.isRunning()) {
				Thread.sleep(count * 10000);
				if (count != 20)
					count++;
			}
			FileOutputStream fos = new FileOutputStream(fName);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(nn);
			oos.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found while storing Neural Net.");
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("IO error while storing Neural Net.");
			e.printStackTrace();
		} catch (InterruptedException e) {
			System.out.format(
					"Putting off saving operation, sleep %d*1000 ms.", count);
		}
	}

	/**
	 * used to restore org.joone.net.NeuralNet from a specific file
	 */
	@SuppressWarnings("finally")
	public NeuralNet restoreNeuralNet(String fName) {
		NeuralNet nn = null;
		try {
			FileInputStream fis;
			fis = new FileInputStream(fName);
			ObjectInputStream ois = new ObjectInputStream(fis);
			nn = (NeuralNet) ois.readObject();
		} catch (FileNotFoundException e) {
			System.out.println("File not found while restoring Neural Net.");
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("IO error while restoring Neural Net.");
			e.printStackTrace();
		} finally {
			return nn;
		}
	}

	public Vector validate(NeuralNet InputNN, double[][] inputArray,
			String columnSelector) {
		NeuralNet nn = InputNN.cloneNet();

		Layer input = nn.getInputLayer();
		input.removeAllInputs();
		MemoryInputSynapse menIS = new MemoryInputSynapse();
		menIS.setFirstRow(1);
		menIS.setAdvancedColumnSelector(columnSelector);
		menIS.setInputArray(inputArray);

		input.addInputSynapse(menIS);

		Layer output = nn.getOutputLayer();
		output.removeAllOutputs();
		MemoryOutputSynapse menOS = new MemoryOutputSynapse();
		output.addOutputSynapse(menOS);

		nn.getMonitor().setTotCicles(1);
		nn.getMonitor().setTrainingPatterns(inputArray.length);
		nn.getMonitor().setLearning(false);
		nn.go();

		nn.stop();
		return menOS.getAllPatterns();
	}

	public double[][] Str2Doubles(String... docs) {
		double[][] result = new double[1][docs.length*138];
		for (int i = 0; i < docs.length; i++) {
			String[] d = docs[i].split(";");
			for (int j = 0; j < 138; j++) {
				result[0][i * 138 + j] = new Double(d[j]);
			}
		}
		return result;
	}
}
