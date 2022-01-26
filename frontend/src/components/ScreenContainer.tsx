import React from 'react';
import { BrowserRouter as Router, Switch, Route, Redirect, useHistory, useLocation } from 'react-router-dom';
import HistoryWrapper from './HistoryWrapper';
import Navbar from './Navbar';
import SampleChoiceWrapper from './SampleChoiceWrapper';
const ScreenContainer: React.FC = () => {

  return (
    <>
      <Router>
        <Switch>
          {[
            { path: '/training', text: '', component: <SampleChoiceWrapper /> },
            { path: '/history', text: '', component: <HistoryWrapper /> },
          ].map(({ path, text, component }) => (
            <Route exact path={path}>
              <div className="h-screen w-full fixed text-cyan-50">
                <Navbar />
                {component}
              </div>
            </Route>
          ))}
        </Switch>
      </Router>
    </>
  );
};

export default ScreenContainer;