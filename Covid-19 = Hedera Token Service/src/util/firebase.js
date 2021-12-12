import firebase from 'firebase';

const firebaseConfig = {
  apiKey: "AIzaSyD1KDtf3PPf5PtYmYIfhnubkMTlHDgJTzI",
  authDomain: "hedera-hash-graph.firebaseapp.com",
  databaseURL: "https://hedera-hash-graph-default-rtdb.firebaseio.com",
  projectId: "hedera-hash-graph",
  storageBucket: "hedera-hash-graph.appspot.com",
  messagingSenderId: "903673590600",
  appId: "1:903673590600:web:9118409fb8028cbd56397d",
  measurementId: "${config.measurementId}"
};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);

export default firebase;
