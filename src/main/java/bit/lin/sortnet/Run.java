package bit.lin.sortnet;

import java.io.File;
import java.util.Vector;

import org.joone.engine.Layer;
import org.joone.engine.Pattern;
import org.joone.io.MemoryInputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;

import bit.lin.utils.JUtils;

public class Run {
	public static void train() {
		SortNet sn = new SortNet(10);
		sn.setCicles(200);
		sn.setPatternCount(500);
		sn.train(true);
		sn.setLearningRate(0.008);
		sn.saveNeuralNet("src/main/resources/naive.nn");
	}

	public static void validate() {
		SortNet sn = SortNet.restoreNeuralNet("src/main/resources/naive.nn");
		NeuralNet nn = sn.getNeuralNet().cloneNet();
		// System.out.println(nn.getMonitor().isUseRMSE());
		Layer input = nn.getInputLayer();
		input.removeAllInputs();
		FilesInputSynapse4SN inputData = new FilesInputSynapse4SN();
		inputData.setAdvancedColumnSelector("3-138,141-276");
		inputData.setBuffered(false);
		inputData.setFirstRow(2);
		inputData
				.setInputFile(
						new File(
								"/home/lins/data/learning to rank/10k4j/Fold1/train.txt"),
						new File(
								"/home/lins/data/learning to rank/10k4j/Fold1/train.txt"));
		input.addInputSynapse(inputData);

		Layer output = nn.getOutputLayer();
		output.removeAllOutputs();
		MemoryOutputSynapse menOS = new MemoryOutputSynapse();
		output.addOutputSynapse(menOS);
		// menOS.

		nn.getMonitor().setTotCicles(1);
		nn.getMonitor().setValidation(true);
		nn.getMonitor().setValidationPatterns(50);
		nn.getMonitor().setLearning(false);
		nn.go();
		nn.stop();

		Vector v = menOS.getAllPatterns();
		for (int i = 0; i < v.size(); i++) {
			Pattern p = (Pattern) v.get(i);
			double[] pattern = p.getArray();
			System.out.println("Output pattern#" + (i/2 + 1) + " : " + pattern[0]
					+ " - " + pattern[1]);
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Run.train();
	}

}
