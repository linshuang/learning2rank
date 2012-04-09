package bit.lin.pairwise.myrank;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Date;

public class Trainer {
	private String _trainSet = "/home/lins/data/learning to rank/10k4j/Fold1/train.txt";

	public String getTrainSet() {
		return _trainSet;
	}

	public void setTrainSet(String trainSet) {
		_trainSet = trainSet;
	}

	PrfIntervalClassifier _pic;

	public Trainer(PrfIntervalClassifier pic) {
		_pic = pic;
	}

	/**
	 * 泡起来，进行训练
	 * 
	 * @param patterns
	 *            a pattern is a query
	 * @param epoches
	 *            just epoch
	 */
	public void run(int patterns, int epoches) {
		try {
			_pic.isStarted = true;
			System.out.format("%s: New epoch started.\n", new Date());

			while (epoches > 0) {
				int pttn = patterns;
				RandomAccessFile raf = new RandomAccessFile(_trainSet, "r");

				while (pttn > 0) {
					ArrayList<String> p = findNextPattern(raf);
					trainWithNN(p);
					pttn--;
					System.out.format("%s: Epoch %d - Patterns %d.\n",
							new Date(), epoches, pttn);
				}

				epoches--;
			}
			System.out.format("%s: Training ended.\n", new Date());
			_pic.isStarted = false;
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * 泡起来，进行训练
	 * 
	 * @param patterns
	 *            a pattern is a query
	 * @param epoches
	 *            just epoch
	 */
	public void run(int patterns, int epoches, int pos) {
		try {
			_pic.isStarted = true;
			System.out.format("%s: New epoch started.\n", new Date());

			while (epoches > 0) {
				RandomAccessFile raf = new RandomAccessFile(_trainSet, "r");
				skip(raf, pos);
				int pttn = patterns;
				while (pttn > 0) {
					ArrayList<String> p = findNextPattern(raf);
					trainWithNN(p);
					System.out.format("%s: Epoch %d - Patterns %d.\n",
							new Date(), epoches, pttn);
					pttn--;
					p = null;
				}
				raf = null;
				epoches--;
			}
			System.out.format("%s: Training ended.\n", new Date());
			_pic.isStarted = false;
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void skip(RandomAccessFile raf, int pos) throws IOException {
		for (; pos > 0; pos--) {
			String line = raf.readLine();
			if (line == null)
				return;
			String qId = line.split(";")[1];

			while (qId.equals(line.split(";")[1])) {
				line = raf.readLine();
				if (line == null)
					break;
			}
			line = null;
		}
	}

	private ArrayList<String> findNextPattern(RandomAccessFile raf)
			throws IOException {
		ArrayList<String> pattern = new ArrayList<String>();

		String line = raf.readLine();
		if (line == null)
			return null;
		String qId = line.split(";")[1];

		while (qId.equals(line.split(";")[1])) {
			pattern.add(line);
			line = raf.readLine();
			if (line == null)
				break;
		}

		return pattern;
	}

	/**
	 * 这种实现是不考虑整体误差的
	 */
	private void trainWithNN(ArrayList<String> ptt) {
		if (ptt == null) {
			return;
		}

		String[] pair = new String[2];
		for (int i = 0; i < ptt.size(); i++) {
			for (int j = 0; j < ptt.size(); j++) {
				if (i == j)
					continue;
				pair[0] = ptt.get(i);
				pair[1] = ptt.get(j);

				// _pic.getSubClassifier()[interval + 5].setInputPatterns(pair);
				// _pic.getSubClassifiers()[interval + 5].train(false);
				for (SubClassifier s : _pic.getSubClassifiers()) {
					// s.setInputPatterns4t(new String[] { pair[0], pair[1] });
					// s.train();
					SubTrainer st = new SubTrainer(s, pair[0], pair[1]);
					st.run();
				}
			}
		}
	}

	/**
	 * 该类用来实现对每一个SubClassifier的训练。每次训练则新建线程。volatile使得每次训练保证数据最新，而不是来自线程本地的寄存器。
	 */
	class SubTrainer extends Thread {
		volatile SubClassifier _s;
		String _doc1;
		String _doc2;

		public SubTrainer(SubClassifier s, String doc1, String doc2) {
			_s = s;
			_doc1 = doc1;
			_doc2 = doc2;
		}

		public void run() {
			_s.setInputPatterns4t(new String[] { _doc1, _doc2 });
			_s.train();
		}
	}

}
