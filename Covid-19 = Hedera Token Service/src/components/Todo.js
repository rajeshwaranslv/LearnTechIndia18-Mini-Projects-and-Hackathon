import React from 'react';
import firebase from '../util/firebase';
import '../App.css';

export default function Todo({ todo }) {
  const deleteTodo = () => {
    const todoRef = firebase.database().ref('Covid-19 Patient Details').child(todo.id);
    todoRef.remove();
  };
  const completeTodo = () => {
    const todoRef = firebase.database().ref('Covid-19 Patient Details').child(todo.id);
    todoRef.update({
      status: !todo.status,
    });
    
  };
  return (
    <div>

 
    
  
 
    </div>
  );
}
