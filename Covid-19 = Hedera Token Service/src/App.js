import React from 'react';
import './App.css';
import Form from './components/Form';
import TodoList from './components/TodoList';
 

export default function App() {
  return (
    <div className="App container-fluid">
      
    
       <div>
       <h1>Hello, Hedera</h1>
        <h1 class="line-1 anim-typewriter"
          style={{
      marginTop:"2rem",
       textShadow:"0 0 3px #666699",
       textTransform:"uppercase",
    
          }}
        >Covid-19 Reporter</h1>

          <Form />
      <TodoList />

      </div>
    </div>
  );
}
