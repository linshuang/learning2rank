package bit.lin.pairwise.sortnet;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Vector;

import org.joone.engine.Layer;
import org.joone.engine.Pattern;
import org.joone.io.FileInputSynapse;
import org.joone.io.MemoryInputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;

import bit.lin.nn.RankResult;
import bit.lin.pointwise.BasicPointwiseNN;
import bit.lin.utils.JUtils;

public class Run {
	static String valiSrc = "/home/lins/data/learning to rank/10k4j/Fold1/vali.txt";
	static String nnLocation = "src/main/resources/sortnet.nn";
	static String serFile = "/home/lins/data/learning to rank/result";

	public static void train(int tPatterns) {
		SortNet sn = new SortNet(10);
		sn.setCicles(700);
		sn.setPatternCount(tPatterns);
		sn.setLearningRate(0.1);
		sn.setBreakErrorChange(0.0000001);
		sn.train(true);
		sn.saveNeuralNet(nnLocation);
	}

	public static void validate() {
		SortNet sn = SortNet.restoreNeuralNet(nnLocation);
		NeuralNet nn = sn.getNeuralNet().cloneNet();
		// System.out.println(nn.getMonitor().isUseRMSE());
		Layer input = nn.getInputLayer();
		input.removeAllInputs();
		FilesInputSynapse4SN inputData = new FilesInputSynapse4SN();
		inputData.setAdvancedColumnSelector("3-138,141-276");
		inputData.setBuffered(false);
		inputData.setFirstRow(1);
		inputData.setInputFile(new File(valiSrc), new File(valiSrc));
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
			System.out.println("Output pattern#" + (i / 2 + 1) + " : "
					+ pattern[0] + " - " + pattern[1]);
		}
	}
	public static void validate2f(int pCount) throws IOException {
		SortNet snn = SortNet.restoreNeuralNet(nnLocation);
		NeuralNet nn = snn.getNeuralNet().cloneNet();
		// System.out.println(nn.getMonitor().isUseRMSE());
		Layer input = nn.getInputLayer();
		input.removeAllInputs();
		FilesInputSynapse4SN inputData = new FilesInputSynapse4SN();
		inputData.setAdvancedColumnSelector("3-138,141-276");
		inputData.setBuffered(false);
		inputData.setFirstRow(1);
		inputData.setInputFile(new File(valiSrc), new File(valiSrc));
		input.addInputSynapse(inputData);

		Layer output = nn.getOutputLayer();
		output.removeAllOutputs();
		MemoryOutputSynapse menOS = new MemoryOutputSynapse();
		output.addOutputSynapse(menOS);
		// menOS.

		nn.getMonitor().setTotCicles(1);
		nn.getMonitor().setValidation(true);
		nn.getMonitor().setValidationPatterns(pCount);
		nn.getMonitor().setLearning(false);
		nn.go();
		nn.stop();

		RandomAccessFile raf = new RandomAccessFile(valiSrc, "r");
		BufferedWriter bw = new BufferedWriter(
				new FileWriter(new File(serFile)));
		Vector v = menOS.getAllPatterns();
		RankResult rRslt = new RankResult();
		String rslt = "";
		String q = "";
		for (int i = 0; i < v.size(); i++) {
			Pattern p = (Pattern) v.get(i);
			double[] pattern = p.getArray();
			String line = raf.readLine();
			String[] tmp = line.split(";");

			if (rRslt.getQ() == -1)
				rRslt.reset(tmp[1]);

			if (rRslt.getQ() != new Integer(line.split(";")[1])) {
				rRslt.sort();
				bw.write(rRslt.getTruthString());
				rRslt.reset(tmp[1]);
			}
			rRslt.addS2T(pattern[0], tmp[0]);
		}
		bw.close();
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Run.validate();
	}

}
