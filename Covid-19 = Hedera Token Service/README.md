# Description

This is a simple todo app with react and firebase realtime database.

## install

```sh
npm install
```

# crud

### create

```js
    const ref = firebase.database().ref("your ref");
    // initialize data
    const data = {....}
    //add to db
    ref.push(data);
```

### delete entire data

```js
const ref = firebase.database().ref('your ref');
ref.remove(data);
```

### delete child data

```js
const child = firebase.database().ref('parent').child(id);
child.remove();
```

### update

```js
    //select which child you want to update
    const child = firebase.database().ref("partent").child(id);
    child.update({
        complete:true,
        ...
    })
```

### read

```js
const ref = firebase.database().ref('parent');

ref.on('value', (snapshot) => {
  console.log(snapshot.val());
});

/* this will listen to the parent if there is something change */

ref.once('value', (snapshot) => {
  console.log(snapshot.val());
});
/* this will listen only one time*/
Required packages for Hedera Token Service:

dotenv: npm install dotenv
hashgraph/SDK: npm install --save hashgraph/SDK

Hosting initiating:

1. npm install firebase

2. Create firebase.json

3. Paste CDN data to firebase.js

4. firebase init hosting

5. Select "create from an existing project"

6. Y N N

7. npm run build 

8. serve -s build

9.  firebase deploy
```
